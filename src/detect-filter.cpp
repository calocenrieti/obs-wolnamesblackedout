#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <cstdint>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>
#include <condition_variable>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "ocr/PaddleOCRRecognizer.h"

#include "yolodetector/YOLODetector.h"

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#endif

static inline void update_running_avg(double &avg, uint64_t &samples, double value)
{
	++samples;
	avg += (value - avg) / static_cast<double>(samples);
}

static uint32_t recommend_num_threads(const std::string &use_gpu)
{
	const bool using_gpu = (use_gpu == "dml" || use_gpu == "cuda");
	if (using_gpu) {
		// Keep CPU-side ORT thread pool small when GPU EP is active.
		return 1;
	}

	const unsigned int hw = std::thread::hardware_concurrency();
	if (hw == 0) {
		return 1;
	}
	if (hw <= 4) {
		return 1;
	}
	if (hw <= 8) {
		return 2;
	}
	if (hw <= 16) {
		return 3;
	}
	return 4;
}

#ifdef _WIN32
static bool ensureGpuInputTexture(filter_data *tf, uint32_t width, uint32_t height)
{
	if (!tf || width == 0 || height == 0) {
		return false;
	}

	if (tf->gpuInputTexture && tf->gpuInputTextureWidth == width && tf->gpuInputTextureHeight == height &&
	    tf->gpuInputSharedHandle != nullptr) {
		return true;
	}

	if (tf->gpuInputTexture) {
		tf->gpuInputTexture->Release();
		tf->gpuInputTexture = nullptr;
	}
	if (tf->gpuInputSharedHandle) {
		CloseHandle(tf->gpuInputSharedHandle);
		tf->gpuInputSharedHandle = nullptr;
	}

	if (!tf->texrender) {
		return false;
	}
	gs_texture_t *obsTexture = gs_texrender_get_texture(tf->texrender);
	if (!obsTexture) {
		return false;
	}
	ID3D11Texture2D *renderTexture = reinterpret_cast<ID3D11Texture2D *>(gs_texture_get_obj(obsTexture));
	if (!renderTexture) {
		return false;
	}

	ID3D11Device *device = nullptr;
	renderTexture->GetDevice(&device);
	if (!device) {
		return false;
	}

	D3D11_TEXTURE2D_DESC desc{};
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

	HRESULT hr = device->CreateTexture2D(&desc, nullptr, &tf->gpuInputTexture);
	if (FAILED(hr) || !tf->gpuInputTexture) {
		device->Release();
		return false;
	}

	IDXGIResource1 *dxgiResource = nullptr;
	hr = tf->gpuInputTexture->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&dxgiResource));
	if (FAILED(hr) || !dxgiResource) {
		tf->gpuInputTexture->Release();
		tf->gpuInputTexture = nullptr;
		device->Release();
		return false;
	}

	hr = dxgiResource->CreateSharedHandle(nullptr,
		DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
		nullptr,
		&tf->gpuInputSharedHandle);
	dxgiResource->Release();
	device->Release();
	if (FAILED(hr) || !tf->gpuInputSharedHandle) {
		tf->gpuInputTexture->Release();
		tf->gpuInputTexture = nullptr;
		return false;
	}

	tf->gpuInputTextureWidth = width;
	tf->gpuInputTextureHeight = height;
	return true;
}

static bool copyGsTextureToD3D11Texture(gs_texture_t *sourceTexture, ID3D11Texture2D *destinationTexture,
					 uint32_t width, uint32_t height)
{
	if (!sourceTexture || !destinationTexture || width == 0 || height == 0) {
		return false;
	}

	ID3D11Texture2D *sourceD3D11Texture = reinterpret_cast<ID3D11Texture2D *>(gs_texture_get_obj(sourceTexture));
	if (!sourceD3D11Texture) {
		return false;
	}

	ID3D11Device *device = nullptr;
	sourceD3D11Texture->GetDevice(&device);
	if (!device) {
		return false;
	}

	ID3D11DeviceContext *context = nullptr;
	device->GetImmediateContext(&context);
	if (!context) {
		device->Release();
		return false;
	}

	D3D11_BOX box{};
	box.left = 0;
	box.top = 0;
	box.front = 0;
	box.right = width;
	box.bottom = height;
	box.back = 1;
	context->CopySubresourceRegion(destinationTexture, 0, 0, 0, 0, sourceD3D11Texture, 0, &box);
	context->Flush();
	context->Release();
	device->Release();
	return true;
}
#endif

static double recommend_ocr_refresh_interval_sec(filter_data *tf)
{
	double ocr_ms_avg = 0.0;
	{
		std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
		ocr_ms_avg = tf->perfOcrMsAvg;
	}

	const bool ocr_on_cpu = !(tf->useGPU == "dml" || tf->useGPU == "cuda");

	if (ocr_on_cpu) {
		// CPU OCR is expensive and usually not worth refreshing every few seconds.
		if (ocr_ms_avg >= 30.0) {
			return 15.0;
		}
		if (ocr_ms_avg >= 20.0) {
			return 10.0;
		}
		if (ocr_ms_avg >= 12.0) {
			return 6.0;
		}
		return 3.0;
	}

	if (ocr_ms_avg >= 30.0) {
		return 5.0;
	}
	if (ocr_ms_avg >= 20.0) {
		return 3.0;
	}
	if (ocr_ms_avg >= 12.0) {
		return 2.0;
	}
	return 1.0;
}

static uint32_t recommend_ocr_max_rois_per_frame(filter_data *tf)
{
	double ocr_ms_avg = 0.0;
	{
		std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
		ocr_ms_avg = tf->perfOcrMsAvg;
	}

	const uint32_t configured_max = std::max<uint32_t>(1, tf->ocrMaxRoisPerFrame);
	if (ocr_ms_avg >= 100.0) {
		return std::min<uint32_t>(configured_max, 1);
	}
	if (ocr_ms_avg >= 70.0) {
		return std::min<uint32_t>(configured_max, 2);
	}
	if (ocr_ms_avg >= 40.0) {
		return std::min<uint32_t>(configured_max, 3);
	}
	if (ocr_ms_avg >= 20.0) {
		return std::min<uint32_t>(configured_max, 4);
	}
	return configured_max;
}

// utility: trim whitespace
static inline std::string trim_copy(const std::string &s)
{
	size_t start = 0;
	while (start < s.size() && isspace((unsigned char)s[start])) start++;
	size_t end = s.size();
	while (end > start && isspace((unsigned char)s[end-1])) end--;
	return s.substr(start, end - start);
}

// split by comma and trim entries
static inline std::vector<std::string> split_comma_list(const std::string &s)
{
	std::vector<std::string> out;
	size_t i = 0;
	while (i < s.size()) {
		size_t j = s.find(',', i);
		if (j == std::string::npos) j = s.size();
		std::string token = trim_copy(s.substr(i, j - i));
		if (!token.empty()) out.push_back(token);
		i = j + 1;
	}
	return out;
}

static std::string sanitize_ocr_text(const std::string &text)
{
	std::string out;
	out.reserve(text.size());
	for (unsigned char c : text) {
		if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == '\'' || c == ' ') {
			out.push_back(c);
		}
	}
	return out;
}

// Levenshtein distance
static int levenshtein_distance(const std::string &a, const std::string &b)
{
	const size_t n = a.size();
	const size_t m = b.size();
	if (n == 0) return (int)m;
	if (m == 0) return (int)n;
	std::vector<int> prev(m + 1), cur(m + 1);
	for (size_t j = 0; j <= m; ++j) prev[j] = (int)j;
	for (size_t i = 1; i <= n; ++i) {
		cur[0] = (int)i;
		for (size_t j = 1; j <= m; ++j) {
			int cost = (a[i-1] == b[j-1]) ? 0 : 1;
			cur[j] = std::min({ prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost });
		}
		prev.swap(cur);
	}
	return prev[m];
}

// similarity ratio based on Levenshtein (0.0 - 1.0)
static double levenshtein_similarity(const std::string &a_in, const std::string &b_in)
{
	std::string a = a_in;
	std::string b = b_in;
	if (a.empty() && b.empty()) return 1.0;
	// normalize to lower for case-insensitive comparison
	std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c){ return std::tolower(c); });
	std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c){ return std::tolower(c); });
	int dist = levenshtein_distance(a, b);
	int maxlen = std::max((int)a.size(), (int)b.size());
	if (maxlen == 0) return 1.0;
	return 1.0 - (double)dist / (double)maxlen;
}

// Check if OCR text partially matches exclude text by comparing parts before/after space
static bool is_partial_match(const std::string &ocr_text, const std::string &exclude_text)
{
	size_t ocr_space = ocr_text.find(' ');
	size_t exclude_space = exclude_text.find(' ');

	// Check first part (before space)
	{
		std::string ocr_first = (ocr_space != std::string::npos) ? ocr_text.substr(0, ocr_space) : ocr_text;
		std::string exclude_first = (exclude_space != std::string::npos) ? exclude_text.substr(0, exclude_space) : exclude_text;

		if (!ocr_first.empty() && !exclude_first.empty()) {
			double ratio = (double)ocr_first.length() / (double)exclude_first.length();
			if (ratio >= 0.7 && exclude_first.find(ocr_first) != std::string::npos) {
				return true;
			}
		}
	}

	// Check second part (after space)
	{
		std::string ocr_second = (ocr_space != std::string::npos) ? ocr_text.substr(ocr_space + 1) : "";
		std::string exclude_second = (exclude_space != std::string::npos) ? exclude_text.substr(exclude_space + 1) : "";

		if (!ocr_second.empty() && !exclude_second.empty()) {
			double ratio = (double)ocr_second.length() / (double)exclude_second.length();
			if (ratio >= 0.7 && exclude_second.find(ocr_second) != std::string::npos) {
				return true;
			}
		}
	}

	return false;
}

struct detect_filter : public filter_data {};

/**
 * @brief Check if a rectangle is completely contained within the exclude range
 * @param rect Detection rectangle (x, y, width, height)
 * @param exclude_left Left boundary of exclude range (pixels from left)
 * @param exclude_right Right boundary of exclude range (pixels from right)
 * @param exclude_top Top boundary of exclude range (pixels from top)
 * @param exclude_bottom Bottom boundary of exclude range (pixels from bottom)
 * @param frame_width Frame width in pixels
 * @param frame_height Frame height in pixels
 * @return true if rect is completely within the exclude range
 */
static bool is_rect_excluded(const cv::Rect_<float>& rect, int exclude_left, int exclude_right,
			     int exclude_top, int exclude_bottom, int frame_width, int frame_height)
{
	// Calculate the excluded area boundaries
	float exclude_area_left = (float)exclude_left;
	float exclude_area_right = (float)(frame_width - exclude_right);
	float exclude_area_top = (float)exclude_top;
	float exclude_area_bottom = (float)(frame_height - exclude_bottom);

	// Check if the rectangle is completely contained within the exclude range
	return rect.x >= exclude_area_left &&
	       rect.x + rect.width <= exclude_area_right &&
	       rect.y >= exclude_area_top &&
	       rect.y + rect.height <= exclude_area_bottom;
}
static void draw_exclude_preview(cv::Mat &frame, const cv::Rect &excludeRect)
{
 	if (excludeRect.width <= 0 || excludeRect.height <= 0) {
 		return;
 	}

 	cv::Mat overlay = frame.clone();
 	overlay.setTo(cv::Scalar(0, 255, 255));
 
 	cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(255));
 	cv::rectangle(mask, excludeRect, cv::Scalar(0), -1);
 
 	cv::Mat shaded = frame.clone();
 	overlay.copyTo(shaded, mask);
 	const double alpha = 0.12;
 	cv::addWeighted(frame, 1.0 - alpha, shaded, alpha, 0, frame);
 
 	drawDashedRectangle(frame, excludeRect, cv::Scalar(0, 255, 255), 2, 8, 15);
}
static void inference_thread_proc(detect_filter *tf)
{
	std::unique_lock<std::mutex> lock(tf->inferenceMutex);

	while (!tf->stopInferenceThread) {
		tf->inferenceCv.wait(lock, [tf] {
			return tf->stopInferenceThread || tf->pendingInferenceFrameReady;
		});

		if (tf->stopInferenceThread) {
			break;
		}

		PendingInferenceFrame frame = std::move(tf->pendingInferenceFrame);
		tf->pendingInferenceFrame.clear();
		tf->pendingInferenceFrameReady = false;
		lock.unlock();

		std::vector<Object> objects;
		std::vector<std::string> classNames;

		try {
			std::unique_lock<std::mutex> modelLock(tf->modelMutex);
			if (tf->yolodetector) {
				const auto yolo_start = std::chrono::steady_clock::now();
				std::optional<std::vector<YOLODetector::BoundingBox>> bboxes_opt;
				bboxes_opt = frame.hasGpuSharedHandle()
					? tf->yolodetector->inferenceFromSharedHandle(frame.gpuSharedHandle, frame.gpuWidth, frame.gpuHeight, tf->conf_threshold)
					: frame.hasBGRA()
					? tf->yolodetector->inferenceBGRA(frame.frameBGRA, tf->conf_threshold)
					: tf->yolodetector->inference(frame.frameBGR, tf->conf_threshold);
				if (frame.hasGpuSharedHandle() && frame.gpuSharedHandle) {
					CloseHandle(frame.gpuSharedHandle);
					frame.gpuSharedHandle = nullptr;
				}
				const auto yolo_end = std::chrono::steady_clock::now();
				const double yolo_ms =
					std::chrono::duration<double, std::milli>(yolo_end - yolo_start).count();
				{
					std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
					update_running_avg(tf->perfYoloMsAvg, tf->perfYoloSamples, yolo_ms);
				}
				if (bboxes_opt.has_value()) {
					objects = tf->yolodetector->convertToObjects(bboxes_opt.value());
					for (const auto &obj : objects) {
						if ((size_t)obj.label >= classNames.size()) {
							classNames.resize(obj.label + 1, "class_" + std::to_string(obj.label));
						}
					}
				}

				// If tracking is enabled, pass detections to ByteTrack and replace objects with tracked results
				if (tf->trackingEnabled) {
					try {
						if (!tf->tracker) {
							// initialize tracker with defaults (frame_rate=30, track_buffer=30)
							tf->tracker = std::make_unique<byte_track::BYTETracker<ByteTrack::Detection, ByteTrack::Track>>(30, 30, tf->conf_threshold);
						}
						// convert to ByteTrack detections
						std::vector<std::shared_ptr<ByteTrack::Detection>> dets;
						dets.reserve(objects.size());
						for (const auto &o : objects) {
							dets.push_back(std::make_shared<ByteTrack::Detection>(o));
						}
						// run tracker
						auto tracked = tf->tracker->update(dets);
						// convert tracked user tracks back to Objects
						std::vector<Object> tracked_objects;
						tracked_objects.reserve(tracked.size());

						std::vector<cv::Rect> ocr_rects;
						std::vector<uint64_t> ocr_track_ids;
					const bool do_ocr = tf->ocrEnabled && tf->ocrRecognizer && tf->ocrRecognizer->isReady();
					auto now = std::chrono::steady_clock::now();
					bool needs_ocr_refresh = false;
					const double refresh_interval_sec = recommend_ocr_refresh_interval_sec(tf);
					const uint32_t max_rois_for_this_frame = recommend_ocr_max_rois_per_frame(tf);
					{
						std::lock_guard<std::mutex> ocrLock(tf->ocrMutex);
						needs_ocr_refresh = do_ocr &&
							(now - tf->lastOcrRefreshTime) >=
							std::chrono::duration<double>(refresh_interval_sec) &&
							!tf->pendingOcrWork;
					}
					for (const auto &ut : tracked) {
						if (!ut) continue;
						const Object &o = ut->getObject();
						tracked_objects.push_back(o);

						if (!tf->ocrEnabled || !tf->ocrRecognizer || !tf->ocrRecognizer->isReady()) {
							continue;
						}

						const uint64_t track_id = o.id;
						if (track_id == 0) {
							continue;
						}

						const bool missing_text = (tf->latestOcrTexts.find(track_id) == tf->latestOcrTexts.end());
						if (!missing_text && !needs_ocr_refresh) {
							continue;
						}

						if (ocr_rects.size() >= max_rois_for_this_frame) {
							continue;
						}

						const int frameWidth = frame.hasBGRA() ? frame.frameBGRA.cols : static_cast<int>(frame.gpuWidth);
						const int frameHeight = frame.hasBGRA() ? frame.frameBGRA.rows : static_cast<int>(frame.gpuHeight);
						if (frameWidth <= 0 || frameHeight <= 0) {
							continue;
						}

						cv::Rect rect = o.rect;
						rect &= cv::Rect(0, 0, frameWidth, frameHeight);
						if (rect.area() <= 0) {
							continue;
						}

						ocr_rects.push_back(rect);
						ocr_track_ids.push_back(track_id);
					}

					if (!ocr_rects.empty()) {
						std::lock_guard<std::mutex> ocrLock(tf->ocrMutex);
						if (!tf->pendingOcrWork) {
							tf->pendingOcrFrameBGRA = frame.hasBGRA() ? frame.frameBGRA : cv::Mat();
							tf->pendingOcrGpuSharedHandle = nullptr;
							tf->pendingOcrGpuWidth = 0;
							tf->pendingOcrGpuHeight = 0;
							if (frame.hasGpuSharedHandle() && frame.gpuSharedHandle) {
								HANDLE duplicatedHandle = nullptr;
								if (DuplicateHandle(GetCurrentProcess(), frame.gpuSharedHandle, GetCurrentProcess(), &duplicatedHandle,
										 0, FALSE, DUPLICATE_SAME_ACCESS)) {
									tf->pendingOcrGpuSharedHandle = duplicatedHandle;
									tf->pendingOcrGpuWidth = frame.gpuWidth;
									tf->pendingOcrGpuHeight = frame.gpuHeight;
								}
							}
							tf->pendingOcrRects = std::move(ocr_rects);
							tf->pendingOcrTrackIds = std::move(ocr_track_ids);
							tf->pendingOcrWork = true;
							tf->ocrCv.notify_one();
						}
					}
					objects = std::move(tracked_objects);
					} catch (const std::exception &e) {
						obs_log(LOG_ERROR, "ByteTrack exception: %s", e.what());
					}
				}
			}
		} catch (const Ort::Exception &e) {
			obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "%s", e.what());
		}

		{
			std::lock_guard<std::mutex> resultsLock(tf->latestObjectsLock);
			tf->latestInferenceObjects = std::move(objects);
			tf->classNames = std::move(classNames);
		}

		tf->inferenceCompleted = true;
		tf->inferenceCv.notify_one();
		lock.lock();
	}
}

static void ocr_thread_proc(detect_filter *tf)
{
	std::unique_lock<std::mutex> lock(tf->ocrMutex);

	while (!tf->stopOcrThread) {
		tf->ocrCv.wait(lock, [tf] {
			return tf->stopOcrThread || tf->pendingOcrWork;
		});

		if (tf->stopOcrThread) {
			break;
		}

		auto ocr_frame_bgra = std::move(tf->pendingOcrFrameBGRA);
		HANDLE ocr_gpu_shared_handle = tf->pendingOcrGpuSharedHandle;
		uint32_t ocr_gpu_width = tf->pendingOcrGpuWidth;
		uint32_t ocr_gpu_height = tf->pendingOcrGpuHeight;
		auto ocr_rects = std::move(tf->pendingOcrRects);
		auto ocr_track_ids = std::move(tf->pendingOcrTrackIds);
		tf->pendingOcrFrameBGRA.release();
		tf->pendingOcrGpuSharedHandle = nullptr;
		tf->pendingOcrGpuWidth = 0;
		tf->pendingOcrGpuHeight = 0;
		tf->pendingOcrRects.clear();
		tf->pendingOcrTrackIds.clear();
		tf->pendingOcrWork = false;
		lock.unlock();

		if (!ocr_rects.empty() && tf->ocrRecognizer) {
			try {
				const double ocr_roi_count = static_cast<double>(ocr_rects.size());
				const auto ocr_start = std::chrono::steady_clock::now();
				auto ocr_results = (ocr_gpu_shared_handle && ocr_gpu_width > 0 && ocr_gpu_height > 0)
					? tf->ocrRecognizer->inferBatch(ocr_gpu_shared_handle, ocr_gpu_width, ocr_gpu_height, ocr_rects, tf->ocrExpandPixels)
					: tf->ocrRecognizer->inferBatch(ocr_frame_bgra, ocr_rects, tf->ocrExpandPixels);
				if (ocr_gpu_shared_handle) {
					CloseHandle(ocr_gpu_shared_handle);
				}
				const auto ocr_end = std::chrono::steady_clock::now();
				const double ocr_ms =
					std::chrono::duration<double, std::milli>(ocr_end - ocr_start).count();
				{
					std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
					update_running_avg(tf->perfOcrMsAvg, tf->perfOcrSamples, ocr_ms);
					update_running_avg(tf->perfOcrRoiCountAvg, tf->perfOcrRoiSamples, ocr_roi_count);
				}
				std::lock_guard<std::mutex> ocrResultsLock(tf->latestObjectsLock);
				for (size_t idx = 0; idx < ocr_results.size() && idx < ocr_track_ids.size(); ++idx) {
					uint64_t tid = ocr_track_ids[idx];
					const std::string new_text = sanitize_ocr_text(ocr_results[idx].text);

					auto it_prev = tf->latestOcrTexts.find(tid);
					bool prev_matched_exclude = false;
					if (it_prev != tf->latestOcrTexts.end()) {
						const std::string prev_text = sanitize_ocr_text(it_prev->second);
						for (const auto &ex : tf->maskExcludeTexts) {
							if (!ex.empty() && prev_text == ex) {
								prev_matched_exclude = true;
								break;
							}
						}
					}

					if (prev_matched_exclude && !tf->maskExcludeTexts.empty()) {
						// If previously matched an exclude entry, and new OCR is similar to
						// any exclude entry (>= ocrContinueThreshold), keep previous text (continue to exclude).
						bool similar_to_exclude = false;
						for (const auto &ex : tf->maskExcludeTexts) {
							if (ex.empty()) continue;
							double sim = levenshtein_similarity(new_text, ex);
							if (sim >= tf->ocrContinueThreshold) {
								similar_to_exclude = true;
								break;
							}
							// Also check partial match by space-separated parts
							if (is_partial_match(new_text, ex)) {
								similar_to_exclude = true;
								break;
							}
						}
						if (similar_to_exclude) {
							// skip updating to preserve exclude state
							continue;
						}
					}

					// default: update OCR text
					tf->latestOcrTexts[tid] = new_text;
				}
				{
					std::lock_guard<std::mutex> refreshLock(tf->ocrMutex);
					tf->lastOcrRefreshTime = std::chrono::steady_clock::now();
				}
			} catch (const std::exception &e) {
				obs_log(LOG_ERROR, "OCR inference exception: %s", e.what());
			}
		}

		lock.lock();
	}
}

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("WoLNamesBlackedOut");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}


void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}


	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	// inference options group
	obs_properties_t *inference_group = obs_properties_create();
	obs_properties_add_group(props, "inference_group", obs_module_text("InferenceGroup"), OBS_GROUP_NORMAL,
				inference_group);
	obs_properties_add_float_slider(inference_group, "threshold", obs_module_text("Threshold"), 0.0,
		1.0, 0.01);
	obs_properties_add_int_slider(inference_group, "inference_interval_frames",
		obs_module_text("InferenceIntervalFrames"), 1, 6, 1);
	obs_properties_add_bool(inference_group, "async_inference", obs_module_text("AsyncInference"));


	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

obs_property_t *masking_type = obs_properties_add_list(
		masking_group, "masking_type", obs_module_text("MaskingType"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(masking_type, obs_module_text("MaskingTypeSolidColor"), "solid_color");
	obs_property_list_add_string(masking_type, obs_module_text("MaskingTypeBlur"), "blur");
	obs_property_list_add_string(masking_type, obs_module_text("MaskingTypePixelate"), "pixelate");
	obs_property_list_add_string(masking_type, obs_module_text("MaskingTypeInpaint"), "inpaint");
	obs_property_list_add_string(masking_type, obs_module_text("MaskingTypeTransparent"), "transparent");

	obs_properties_add_color(masking_group, "masking_color",
				obs_module_text("MaskingColor"));
	obs_properties_add_int_slider(masking_group, "masking_blur_radius",
				obs_module_text("MaskingBlurRadius"), 0, 100, 1);
	obs_properties_add_int_slider(masking_group, "dilation_iterations",
				obs_module_text("DilationIterations"), 0, 50, 1);
	obs_properties_add_int_slider(masking_group, "inpaint_radius",
				obs_module_text("InpaintRadius"), 1, 200, 1);

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								obs_property_t *,
								obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_t *masking_type = obs_properties_get(props_, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_inpaint_radius =
			obs_properties_get(props_, "inpaint_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");

		obs_property_set_visible(masking_type, enabled);
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		obs_property_set_visible(masking_inpaint_radius, false);
		obs_property_set_visible(masking_dilation, enabled);
		return true;
	});

 	// add callback to show/hide blur radius, inpaint radius and async inference
 	obs_property_set_modified_callback(masking_type, [](obs_properties_t *props_,
 					obs_property_t *,
 					obs_data_t *settings) {
 		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
 		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
 		obs_property_t *masking_blur_radius =
 			obs_properties_get(props_, "masking_blur_radius");
 		obs_property_t *masking_inpaint_radius =
 			obs_properties_get(props_, "inpaint_radius");
 		obs_property_t *masking_dilation =
 			obs_properties_get(props_, "dilation_iterations");
 		obs_property_set_visible(masking_color, false);
 		obs_property_set_visible(masking_blur_radius, false);
 		obs_property_set_visible(masking_inpaint_radius, false);
 		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
 		obs_property_set_visible(masking_dilation, masking_enabled);
 		if (masking_type_value == "solid_color") {
 			obs_property_set_visible(masking_color, masking_enabled);
 		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
 			obs_property_set_visible(masking_blur_radius, masking_enabled);
 		} else if (masking_type_value == "inpaint") {
 			obs_property_set_visible(masking_inpaint_radius, masking_enabled);
 		}
 		return true;
 	});

 	// name-based exclusion group
 	obs_properties_t *exclude_by_name_group = obs_properties_create();
 	obs_property_t *exclude_by_name_group_prop =
 		obs_properties_add_group(props, "exclude_by_name_group",
 				obs_module_text("ExcludeByNameGroup"), OBS_GROUP_CHECKABLE, exclude_by_name_group);
	obs_property_t *ocr_expand_pixels = obs_properties_add_int_slider(
		exclude_by_name_group, "ocr_expand_pixels",
		obs_module_text("OCRExpandPixels"), 0, 5, 1);
	obs_property_set_visible(ocr_expand_pixels, false);
	obs_property_t *ocr_max_rois = obs_properties_add_int_slider(
		exclude_by_name_group, "ocr_max_rois",
		obs_module_text("OCRMaxRoisPerFrame"), 1, 32, 1);
	obs_property_set_visible(ocr_max_rois, false);
	obs_property_t *ocr_initial_threshold = obs_properties_add_int_slider(
		exclude_by_name_group, "ocr_initial_threshold",
		obs_module_text("OCRInitialThreshold"), 50, 100, 1);
	obs_property_int_set_suffix(ocr_initial_threshold, "%");
	obs_property_set_visible(ocr_initial_threshold, false);
	obs_property_set_modified_callback(exclude_by_name_group_prop,
		[](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
			const bool enabled = obs_data_get_bool(settings, "exclude_by_name_group");
			obs_data_set_bool(settings, "tracking_group", enabled);
			obs_data_set_bool(settings, "ocr_enabled", enabled);
			obs_property_set_visible(obs_properties_get(props_, "ocr_expand_pixels"), enabled);
			obs_property_set_visible(obs_properties_get(props_, "ocr_max_rois"), enabled);
			obs_property_set_visible(obs_properties_get(props_, "ocr_initial_threshold"), enabled);
			obs_property_set_visible(obs_properties_get(props_, "mask_exclude_text"), enabled);
			return true;
		});

	obs_properties_add_text(exclude_by_name_group, "mask_exclude_text",
		obs_module_text("MaskExcludeText"), OBS_TEXT_DEFAULT);

	// Exclude range group for detection exclusion area
	obs_properties_t *exclude_group = obs_properties_create();
	obs_property_t *exclude_group_prop =
		obs_properties_add_group(props, "exclude_group", obs_module_text("ExcludeGroup"),
					 OBS_GROUP_CHECKABLE, exclude_group);

	// add callback to show/hide exclude range options
	obs_property_set_modified_callback(exclude_group_prop, [](obs_properties_t *props_,
						 obs_property_t *,
						 obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "exclude_group");
		obs_property_t *exclude_preview = obs_properties_get(props_, "exclude_preview");
		obs_property_t *exclude_left = obs_properties_get(props_, "exclude_left");
		obs_property_t *exclude_right = obs_properties_get(props_, "exclude_right");
		obs_property_t *exclude_top = obs_properties_get(props_, "exclude_top");
		obs_property_t *exclude_bottom = obs_properties_get(props_, "exclude_bottom");

		obs_property_set_visible(exclude_preview, enabled);
		obs_property_set_visible(exclude_left, enabled);
		obs_property_set_visible(exclude_right, enabled);
		obs_property_set_visible(exclude_top, enabled);
		obs_property_set_visible(exclude_bottom, enabled);
		return true;
	});

	// add exclude preview toggle
	obs_properties_add_bool(exclude_group, "exclude_preview", obs_module_text("ExcludePreview"));

 	// determine slider limits from source resolution
 	int source_width = 1920;
 	int source_height = 1080;
 	if (tf && tf->source) {
 		source_width = (int)obs_source_get_base_width(tf->source);
 		source_height = (int)obs_source_get_base_height(tf->source);
 	}
 	if (source_width <= 0) {
 		source_width = 1920;
 	}
 	if (source_height <= 0) {
 		source_height = 1080;
 	}

 	// add sliders for exclude range (left, right, top, bottom)
 	obs_properties_add_int_slider(exclude_group, "exclude_left",
 				      obs_module_text("ExcludeLeft"), 0, source_width, 1);
 	obs_properties_add_int_slider(exclude_group, "exclude_right",
 				      obs_module_text("ExcludeRight"), 0, source_width, 1);
 	obs_properties_add_int_slider(exclude_group, "exclude_top",
 				      obs_module_text("ExcludeTop"), 0, source_height, 1);
 	obs_properties_add_int_slider(exclude_group, "exclude_bottom",
 				      obs_module_text("ExcludeBottom"), 0, source_height, 1);

	// Advanced settings group should remain at the bottom.
	obs_properties_t *advanced_group = obs_properties_create();
	obs_properties_add_group(props, "advanced_settings", obs_module_text("AdvancedSettingsGroup"), OBS_GROUP_NORMAL,
				advanced_group);
	obs_properties_add_bool(advanced_group, "preview", obs_module_text("Preview"));
	// obs_properties_add_bool(advanced_group, "gpu_zero_copy", "GPU Zero-Copy Input (experimental)");
	obs_property_t *perf_log = obs_properties_add_bool(advanced_group, "perf_log", "Perf Log");
	obs_property_t *perf_log_interval =
		obs_properties_add_int_slider(advanced_group, "perf_log_interval",
			"Perf Log Interval (frames)", 30, 600, 30);
	obs_property_set_modified_callback(perf_log,
		[](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
			const bool enabled = obs_data_get_bool(settings, "perf_log");
			obs_property_t *interval = obs_properties_get(props_, "perf_log_interval");
			obs_property_set_visible(interval, enabled);
			return true;
		});
	obs_property_set_visible(perf_log_interval, false);

	// Add a informative text about the plugin
 	std::string basic_info =
 		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
 	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

 	UNUSED_PARAMETER(data);
 	return props;
 }

void detect_filter_defaults(obs_data_t *settings)
{
	// obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
		obs_data_set_default_bool(settings, "perf_log", false);
		obs_data_set_default_int(settings, "perf_log_interval", 120);
	obs_data_set_default_double(settings, "threshold", 0.15);
	obs_data_set_default_string(settings, "model_size", "yolodetector");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_bool(settings, "masking_group", true);
	obs_data_set_default_string(settings, "masking_type", "solid_color");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 3);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_bool(settings, "exclude_by_name_group", false);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_bool(settings, "ocr_enabled", false);
	obs_data_set_default_string(settings, "ocr_model_path", "");
	obs_data_set_default_string(settings, "ocr_dict_path", "");
	obs_data_set_default_string(settings, "mask_exclude_text", "");
	obs_data_set_default_int(settings, "ocr_expand_pixels", 0);
	obs_data_set_default_int(settings, "ocr_max_rois", 6);
	obs_data_set_default_int(settings, "ocr_initial_threshold", 80);

 	// Exclude range defaults
 	obs_data_set_default_bool(settings, "exclude_group", false);
 	obs_data_set_default_bool(settings, "exclude_preview", true);
 	obs_data_set_default_int(settings, "exclude_left", 0);
 	obs_data_set_default_int(settings, "exclude_right", 0);
 	obs_data_set_default_int(settings, "exclude_top", 0);
 	obs_data_set_default_int(settings, "exclude_bottom", 0);

 	// Inpaint effect defaults
 	obs_data_set_default_int(settings, "inpaint_radius", 70);

 	// Asynchronous inference default
 	obs_data_set_default_bool(settings, "async_inference", true);
	obs_data_set_default_int(settings, "inference_interval_frames", 1);
	obs_data_set_default_bool(settings, "gpu_zero_copy", true);
 }

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->perfLogEnabled = obs_data_get_bool(settings, "perf_log");
	tf->perfLogInterval = (uint32_t)obs_data_get_int(settings, "perf_log_interval");
	if (tf->perfLogInterval < 30) {
		tf->perfLogInterval = 30;
	}
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	const bool exclude_by_name_enabled = obs_data_get_bool(settings, "exclude_by_name_group");
	tf->trackingEnabled = exclude_by_name_enabled;
	tf->ocrEnabled = exclude_by_name_enabled;
	{
		char *ocrModelPathPtr = obs_module_file("models/c_ppocr-v5-rec_sim.onnx");
		if (ocrModelPathPtr) {
			tf->ocrModelFilepath = ocrModelPathPtr;
			bfree(ocrModelPathPtr);
		} else {
			tf->ocrModelFilepath.clear();
			obs_log(LOG_ERROR, "Failed to resolve OCR model path via obs_module_file");
		}
	}
	{
		char *ocrDictPathPtr = obs_module_file("dict/ppocrv5_en_dict.txt");
		if (ocrDictPathPtr) {
			tf->ocrDictFilepath = ocrDictPathPtr;
			bfree(ocrDictPathPtr);
		} else {
			tf->ocrDictFilepath.clear();
			obs_log(LOG_ERROR, "Failed to resolve OCR dictionary path via obs_module_file");
		}
	}
	tf->ocrExpandPixels = (int)obs_data_get_int(settings, "ocr_expand_pixels");
	tf->ocrMaxRoisPerFrame = (uint32_t)obs_data_get_int(settings, "ocr_max_rois");
	if (tf->ocrMaxRoisPerFrame < 1) {
		tf->ocrMaxRoisPerFrame = 1;
	}
	tf->ocrInitialThreshold = (float)obs_data_get_int(settings, "ocr_initial_threshold") / 100.0f;
	tf->ocrContinueThreshold = tf->ocrInitialThreshold - 0.1f;
	if (!tf->ocrEnabled) {
		tf->latestOcrTexts.clear();
	}
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "dilation_iterations");
	// read raw comma-separated string and parse into trimmed list
	tf->maskExcludeText = obs_data_get_string(settings, "mask_exclude_text");
	tf->maskExcludeTexts = split_comma_list(tf->maskExcludeText);

 	tf->exclude_group_enabled = obs_data_get_bool(settings, "exclude_group");
 	tf->exclude_preview = obs_data_get_bool(settings, "exclude_preview");
 	tf->exclude_left = (int)obs_data_get_int(settings, "exclude_left");
 	tf->exclude_right = (int)obs_data_get_int(settings, "exclude_right");
 	tf->exclude_top = (int)obs_data_get_int(settings, "exclude_top");
 	tf->exclude_bottom = (int)obs_data_get_int(settings, "exclude_bottom");

 	// Inpaint parameters
 	tf->inpaintRadius = (float)obs_data_get_int(settings, "inpaint_radius");

 	// Asynchronous inference setting
 	tf->asyncInference = obs_data_get_bool(settings, "async_inference");
	tf->inferenceIntervalFrames =
		(uint32_t)obs_data_get_int(settings, "inference_interval_frames");
	if (tf->inferenceIntervalFrames < 1) {
		tf->inferenceIntervalFrames = 1;
	}
	tf->gpuZeroCopyEnabled = obs_data_get_bool(settings, "gpu_zero_copy");
	if (tf->yolodetector) {
		tf->yolodetector->setGpuZeroCopyEnabled(tf->gpuZeroCopyEnabled);
	}
	obs_log(LOG_INFO, "  GPU Zero-Copy Input: %s",
		tf->gpuZeroCopyEnabled ? "true" : "false");

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = recommend_num_threads(newUseGpu);
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;

		if (newModelSize == "yolodetector") {
			modelFilepath_rawPtr = obs_module_file("models/my_yolov8m_s.onnx");
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;


		// Load model
		try {

			if (tf->modelSize == "yolodetector") {
				// Initialize YOLODetector for yolodetector model size
				if (!tf->yolodetector) {
					tf->yolodetector = std::make_unique<YOLODetector>();
				}

				// GPU 使用設定とスレッド数設定を適用
				bool use_gpu = (tf->useGPU == "dml" || tf->useGPU == "cuda");
				tf->yolodetector->setUseGPU(use_gpu);
				tf->yolodetector->setGpuZeroCopyEnabled(tf->gpuZeroCopyEnabled);
				tf->yolodetector->setNumThreads(tf->numThreads);

				if (use_gpu) {
					// DirectML 初期化を試行（Windows のみ）
					if (!tf->yolodetector->initializeDirectML()) {
						obs_log(LOG_WARNING, "Failed to initialize DirectML, falling back to CPU");
						tf->yolodetector->setUseGPU(false);
					}
				}
				if (!tf->yolodetector->loadModel(tf->modelFilepath.c_str())) {
					throw std::runtime_error("Failed to load YOLODetector model");
				}

			}

			if (tf->ocrEnabled) {
				if (!tf->ocrRecognizer) {
					tf->ocrRecognizer = std::make_unique<ocr::PaddleOCRRecognizer>();
				}
				tf->ocrRecognizer->setUseDirectML(tf->useGPU == "dml" || tf->useGPU == "cuda");
				tf->ocrRecognizer->setGpuZeroCopyEnabled(tf->gpuZeroCopyEnabled);
				tf->ocrRecognizer->setNumThreads(tf->numThreads);
				if (tf->ocrDictFilepath.empty()) {
					obs_log(LOG_ERROR, "OCR dictionary path is empty");
					throw std::runtime_error("Failed to load OCR dictionary");
				}
				if (!tf->ocrRecognizer->loadDictionary(tf->ocrDictFilepath)) {
					obs_log(LOG_ERROR, "Failed to load OCR dictionary from %s", tf->ocrDictFilepath.c_str());
					throw std::runtime_error("Failed to load OCR dictionary");
				}
				if (tf->ocrModelFilepath.empty()) {
					obs_log(LOG_ERROR, "OCR model path is empty");
					throw std::runtime_error("Failed to load OCR model");
				}
				if (!tf->ocrRecognizer->loadModel(tf->ocrModelFilepath)) {
					obs_log(LOG_ERROR, "Failed to load OCR model from %s", tf->ocrModelFilepath.c_str());
					throw std::runtime_error("Failed to load OCR model");
				}
			}
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());

			return;
		}
	}



	// enable
	tf->isDisabled = false;

	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Hardware Threads: %u", std::thread::hardware_concurrency());
		obs_log(LOG_INFO, "  Num Threads Mode: auto");
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  GPU Zero-Copy Input: %s", tf->gpuZeroCopyEnabled ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Masking Enabled: %s",
			obs_data_get_bool(settings, "masking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Masking Type: %s",
			obs_data_get_string(settings, "masking_type"));
		obs_log(LOG_INFO, "  Masking Color: %s",
			obs_data_get_string(settings, "masking_color"));
		obs_log(LOG_INFO, "  Masking Blur Radius: %d",
			obs_data_get_int(settings, "masking_blur_radius"));
		obs_log(LOG_INFO, "  Name-based exclusion enabled: %s",
			obs_data_get_bool(settings, "exclude_by_name_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Processing enabled: %s", tf->isDisabled ? "false" : "true");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

 	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
 		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
 		{MASKING_EFFECT_PATH, &tf->maskingEffect},
 		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
 		{INPAINT_EFFECT_PATH, &tf->inpaintEffect},
 	};

	for (auto [effectPath, effect] : effects) {
		char *effectPathPtr = obs_module_file(effectPath);
		if (!effectPathPtr) {
			obs_log(LOG_ERROR, "Failed to get effect path: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_enter_graphics();
		*effect = gs_effect_create_from_file(effectPathPtr, nullptr);
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_leave_graphics();
	}

	detect_filter_update(tf, settings);

	// Start asynchronous inference thread once the filter is created.
	tf->stopInferenceThread = false;
	tf->pendingOcrWork = false;
	tf->stopOcrThread = false;
	tf->inferenceThread = std::thread(inference_thread_proc, tf);
	tf->ocrThread = std::thread(ocr_thread_proc, tf);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		{
			std::lock_guard<std::mutex> lock(tf->inferenceMutex);
			tf->stopInferenceThread = true;
			if (tf->latestGpuSharedHandle) {
				CloseHandle(tf->latestGpuSharedHandle);
				tf->latestGpuSharedHandle = nullptr;
			}
			tf->latestGpuWidth = 0;
			tf->latestGpuHeight = 0;
		}
		tf->inferenceCv.notify_one();
		if (tf->inferenceThread.joinable()) {
			tf->inferenceThread.join();
		}

		{
			std::lock_guard<std::mutex> lock(tf->ocrMutex);
			tf->stopOcrThread = true;
			if (tf->pendingOcrGpuSharedHandle) {
				CloseHandle(tf->pendingOcrGpuSharedHandle);
				tf->pendingOcrGpuSharedHandle = nullptr;
			}
			tf->pendingOcrGpuWidth = 0;
			tf->pendingOcrGpuHeight = 0;
		}
		tf->ocrCv.notify_one();
		if (tf->ocrThread.joinable()) {
			tf->ocrThread.join();
		}

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		if (tf->previewUploadTexture) {
			gs_texture_destroy(tf->previewUploadTexture);
			tf->previewUploadTexture = nullptr;
		}
		if (tf->maskUploadTexture) {
			gs_texture_destroy(tf->maskUploadTexture);
			tf->maskUploadTexture = nullptr;
		}
		if (tf->effectWorkTexture) {
			gs_texture_destroy(tf->effectWorkTexture);
			tf->effectWorkTexture = nullptr;
			tf->effectWorkTextureWidth = 0;
			tf->effectWorkTextureHeight = 0;
		}
 		gs_effect_destroy(tf->kawaseBlurEffect);
 		gs_effect_destroy(tf->maskingEffect);
 		gs_effect_destroy(tf->pixelateEffect);
 		gs_effect_destroy(tf->inpaintEffect);
 		obs_leave_graphics();
		if (tf->gpuInputTexture) {
			tf->gpuInputTexture->Release();
			tf->gpuInputTexture = nullptr;
		}
		if (tf->gpuInputSharedHandle) {
			CloseHandle(tf->gpuInputSharedHandle);
			tf->gpuInputSharedHandle = nullptr;
		}
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	const auto tick_start = std::chrono::steady_clock::now();

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	// Check if either model is available
	if (tf->isDisabled || !tf->yolodetector) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	bool hasCpuFrame = false;
	const auto input_copy_start = std::chrono::steady_clock::now();
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (lock.owns_lock() && !tf->inputBGRA.empty()) {
			imageBGRA = tf->inputBGRA;
			hasCpuFrame = true;
		}
	}
	const auto input_copy_end = std::chrono::steady_clock::now();

	PendingInferenceFrame inferenceFrame;
	if (hasCpuFrame) {
		inferenceFrame.frameBGRA = imageBGRA;
	}
	if (tf->gpuZeroCopyEnabled) {
		std::lock_guard<std::mutex> gpuLock(tf->gpuFrameLock);
		if (tf->latestGpuSharedHandle && tf->latestGpuWidth > 0 && tf->latestGpuHeight > 0) {
			HANDLE duplicatedHandle = nullptr;
			if (DuplicateHandle(GetCurrentProcess(), tf->latestGpuSharedHandle, GetCurrentProcess(), &duplicatedHandle,
					 0, FALSE, DUPLICATE_SAME_ACCESS)) {
				inferenceFrame.gpuSharedHandle = duplicatedHandle;
				inferenceFrame.gpuWidth = tf->latestGpuWidth;
				inferenceFrame.gpuHeight = tf->latestGpuHeight;
			}
		}
	}
	if (!hasCpuFrame && !inferenceFrame.hasGpuSharedHandle()) {
		return;
	}
	const bool needsBGRFrame = tf->preview || tf->maskingEnabled || tf->trackingEnabled ||
		(tf->exclude_group_enabled && tf->exclude_preview);
	const auto color_convert_start = std::chrono::steady_clock::now();
	if (needsBGRFrame && hasCpuFrame) {
		cv::cvtColor(imageBGRA, inferenceFrame.frameBGR, cv::COLOR_BGRA2BGR);
	}
	const auto color_convert_end = std::chrono::steady_clock::now();
	cv::Mat previewFrameBGR;
	if (needsBGRFrame) {
		previewFrameBGR = inferenceFrame.frameBGR;
	}

	bool shouldQueueInference = true;
	bool dueByInterval = true;
	{
		std::lock_guard<std::mutex> lock(tf->inferenceMutex);
		tf->inferenceIntervalCounter++;
		if (tf->inferenceIntervalCounter < tf->inferenceIntervalFrames) {
			dueByInterval = false;
		} else {
			tf->inferenceIntervalCounter = 0;
		}

		if (!dueByInterval) {
			shouldQueueInference = false;
		} else {
			tf->pendingInferenceFrame.clear();
			tf->pendingInferenceFrame = std::move(inferenceFrame);
			tf->pendingInferenceFrameReady = true;
			tf->inferenceCompleted = false;
		}
	}
	if (shouldQueueInference) {
		tf->inferenceCv.notify_one();
	}

	// If synchronous mode, wait for inference to complete
	if (!tf->asyncInference && shouldQueueInference) {
		std::unique_lock<std::mutex> lock(tf->inferenceMutex);
		tf->inferenceCv.wait(lock, [tf] {
			return tf->inferenceCompleted || tf->stopInferenceThread;
		});
	}

	std::vector<Object> objects;
	std::vector<std::string> classNames;
	{
		std::lock_guard<std::mutex> lock(tf->latestObjectsLock);
		objects = tf->latestInferenceObjects;
		classNames = tf->classNames;
	}

	// update the detected object text input for YOLODetector with index-based class names
	if (objects.size() > 0) {
		int currentLabel = objects[0].label;
		if (tf->lastDetectedObjectId != currentLabel) {
			tf->lastDetectedObjectId = currentLabel;
			std::string className = "class_" + std::to_string(currentLabel);
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			if (currentLabel < (int)classNames.size() && !classNames[currentLabel].empty()) {
				className = classNames[currentLabel];
			}
			obs_data_set_string(source_settings, "detected_object", className.c_str());
			// release the source settings
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->objectCategory != -1) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.label == tf->objectCategory) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	const bool needsPreviewFrame = tf->preview || (tf->exclude_group_enabled && tf->exclude_preview);
	if (needsPreviewFrame || tf->maskingEnabled) {
		cv::Mat frame;
		if (needsPreviewFrame) {
			if (!hasCpuFrame) {
				return;
			}
			frame = previewFrameBGR.empty() ? cv::Mat() : previewFrameBGR.clone();
			if (frame.empty()) {
				cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);
			}
		}

		const int maskWidth = hasCpuFrame ? imageBGRA.cols : static_cast<int>(inferenceFrame.gpuWidth);
		const int maskHeight = hasCpuFrame ? imageBGRA.rows : static_cast<int>(inferenceFrame.gpuHeight);
		if (tf->maskingEnabled && (maskWidth <= 0 || maskHeight <= 0)) {
			return;
		}

		cv::Mat nextMask;

		if (tf->preview && objects.size() > 0) {
			draw_objects(frame, objects, classNames);
			if (tf->ocrEnabled) {
				std::lock_guard<std::mutex> ocrLock(tf->latestObjectsLock);
				for (const auto &obj : objects) {
					auto it = tf->latestOcrTexts.find(obj.id);
					if (it == tf->latestOcrTexts.end() || it->second.empty()) {
						continue;
					}
					const std::string &ocr_text = it->second;
					int font_face = cv::FONT_HERSHEY_SIMPLEX;
					double font_scale = 0.5;
					int thickness = 1;
					int baseline = 0;
					std::string overlay_text = "ID" + std::to_string(obj.id) + ": " + it->second;
					cv::Size text_size = cv::getTextSize(overlay_text, font_face, font_scale, thickness, &baseline);
					int text_x = std::max(0, (int)obj.rect.x);
					int text_top = std::max(0, (int)obj.rect.y - (text_size.height + baseline + 8));
					cv::rectangle(frame,
						cv::Point(text_x, text_top),
						cv::Point(text_x + text_size.width, text_top + text_size.height + baseline + 6),
						cv::Scalar(0, 0, 0), cv::FILLED);
					cv::Point text_origin(text_x, text_top + text_size.height + 2);
					cv::putText(frame, overlay_text, text_origin, font_face, font_scale,
						cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
				}
			}
		}

		if (tf->exclude_group_enabled && tf->exclude_preview) {
			cv::Rect excludeRect(
				tf->exclude_left,
				tf->exclude_top,
				frame.cols - tf->exclude_left - tf->exclude_right,
				frame.rows - tf->exclude_top - tf->exclude_bottom);
			draw_exclude_preview(frame, excludeRect);
		}
		const auto mask_build_start = std::chrono::steady_clock::now();
		if (tf->maskingEnabled) {
			nextMask = cv::Mat::zeros(cv::Size(maskWidth, maskHeight), CV_8UC1);
			for (const Object &obj : objects) {
				// Check if this detection should be excluded from masking
				if (tf->exclude_group_enabled && is_rect_excluded(obj.rect, tf->exclude_left,
									tf->exclude_right, tf->exclude_top, tf->exclude_bottom,
									maskWidth, maskHeight)) {
					continue;  // Skip this detection - don't add to mask
				}
				if (!tf->maskExcludeTexts.empty() && tf->ocrEnabled) {
					auto it = tf->latestOcrTexts.find(obj.id);
					if (it != tf->latestOcrTexts.end()) {
						const std::string &ocr_text = it->second;
						bool excluded = false;
						for (const auto &ex : tf->maskExcludeTexts) {
							if (ex.empty()) continue;
							// exact match
							if (ocr_text == ex) {
								excluded = true;
								break;
							}
							// Levenshtein similarity check (0.0 - 1.0), treat >= ocrInitialThreshold as match
							double sim = levenshtein_similarity(ocr_text, ex);
							if (sim >= tf->ocrInitialThreshold) {
								excluded = true;
								break;
							}
						}
						if (excluded) {
							continue; // Skip masking for this object when OCR text matches any exclude entry
						}
					}
				}
				cv::rectangle(nextMask, obj.rect, cv::Scalar(255), -1);
			}
			if (tf->maskingDilateIterations > 0) {
				cv::Mat dilatedMask;
				cv::dilate(nextMask, dilatedMask, cv::Mat(),
					   cv::Point(-1, -1), tf->maskingDilateIterations);
				nextMask = std::move(dilatedMask);
			}
		}
		const auto mask_build_end = std::chrono::steady_clock::now();

		cv::Mat nextPreviewBGRA;
		if (needsPreviewFrame) {
			cv::cvtColor(frame, nextPreviewBGRA, cv::COLOR_BGR2BGRA);
		}

		const auto publish_start = std::chrono::steady_clock::now();
		{
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (needsPreviewFrame) {
				tf->outputPreviewBGRA = std::move(nextPreviewBGRA);
			}
			if (tf->maskingEnabled) {
				tf->outputMask = std::move(nextMask);
			}
		}
		const auto publish_end = std::chrono::steady_clock::now();

		std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
		update_running_avg(
			tf->perfMaskBuildMsAvg,
			tf->perfMaskBuildSamples,
			std::chrono::duration<double, std::milli>(mask_build_end - mask_build_start).count());
		update_running_avg(
			tf->perfPublishMsAvg,
			tf->perfPublishSamples,
			std::chrono::duration<double, std::milli>(publish_end - publish_start).count());
	}

	const auto tick_end = std::chrono::steady_clock::now();
	const double tick_ms = std::chrono::duration<double, std::milli>(tick_end - tick_start).count();
	{
		std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
		update_running_avg(
			tf->perfInputCopyMsAvg,
			tf->perfInputCopySamples,
			std::chrono::duration<double, std::milli>(input_copy_end - input_copy_start).count());
		update_running_avg(
			tf->perfColorConvertMsAvg,
			tf->perfColorConvertSamples,
			std::chrono::duration<double, std::milli>(color_convert_end - color_convert_start)
				.count());
		update_running_avg(tf->perfTickMsAvg, tf->perfTickSamples, tick_ms);
		if (tf->perfLogEnabled) {
			tf->perfLogCounter++;
			if (tf->perfLogCounter >= tf->perfLogInterval) {
				tf->perfLogCounter = 0;
				const double obs_fps = obs_get_active_fps();
				obs_log(LOG_INFO,
					"[Perf] fps=%.2f tick=%.2f yolo=%.2f ocr=%.2f ocrroi=%.2f in=%.2f cvt=%.2f mask=%.2f pub=%.2f cap=%.2f snap=%.2f up=%.2f fx=%.2f texcap=%llu stagefb=%llu",
					obs_fps,
					tf->perfTickMsAvg,
					tf->perfYoloMsAvg,
					tf->perfOcrMsAvg,
					tf->perfOcrRoiCountAvg,
					tf->perfInputCopyMsAvg,
					tf->perfColorConvertMsAvg,
					tf->perfMaskBuildMsAvg,
					tf->perfPublishMsAvg,
					tf->perfRenderCaptureMsAvg,
					tf->perfRenderSnapshotMsAvg,
					tf->perfRenderUploadMsAvg,
					tf->perfRenderEffectMsAvg,
					static_cast<unsigned long long>(tf->perfRenderTextureCaptureCount),
					static_cast<unsigned long long>(tf->perfStageSurfaceFallbackCount));
			}
		}
	}

}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->yolodetector) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	const auto render_capture_start = std::chrono::steady_clock::now();
	const bool needsCpuFrame = tf->preview || tf->maskingEnabled || tf->trackingEnabled ||
		(tf->exclude_group_enabled && tf->exclude_preview);
	const bool usedRenderTexture = getRGBAFromRenderTexture(tf, width, height, needsCpuFrame);
	if (!usedRenderTexture && needsCpuFrame && !getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}
	const auto render_capture_end = std::chrono::steady_clock::now();
	{
		std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
		if (usedRenderTexture) {
			tf->perfRenderTextureCaptureCount++;
		} else {
			tf->perfStageSurfaceFallbackCount++;
		}
		update_running_avg(
			tf->perfRenderCaptureMsAvg,
			tf->perfRenderCaptureSamples,
			std::chrono::duration<double, std::milli>(render_capture_end - render_capture_start)
				.count());
	}

	if (tf->gpuZeroCopyEnabled && usedRenderTexture && ensureGpuInputTexture(tf, width, height)) {
		gs_texture_t *renderedTexture = gs_texrender_get_texture(tf->texrender);
		if (renderedTexture && copyGsTextureToD3D11Texture(renderedTexture, tf->gpuInputTexture, width, height)) {
			HANDLE duplicatedHandle = nullptr;
			if (DuplicateHandle(GetCurrentProcess(), tf->gpuInputSharedHandle, GetCurrentProcess(), &duplicatedHandle,
					 0, FALSE, DUPLICATE_SAME_ACCESS)) {
				std::lock_guard<std::mutex> gpuLock(tf->gpuFrameLock);
				if (tf->latestGpuSharedHandle) {
					CloseHandle(tf->latestGpuSharedHandle);
				}
				tf->latestGpuSharedHandle = duplicatedHandle;
				tf->latestGpuWidth = width;
				tf->latestGpuHeight = height;
			}
		}
	}

	// if preview or masking is enabled, render the image
	if (tf->preview || tf->maskingEnabled || (tf->exclude_group_enabled && tf->exclude_preview)) {
		const bool needsPreviewImage = tf->preview || (tf->exclude_group_enabled && tf->exclude_preview);
		cv::Mat outputBGRA, outputMask;
		const auto render_snapshot_start = std::chrono::steady_clock::now();
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (needsPreviewImage) {
				if (tf->outputPreviewBGRA.empty()) {
					std::lock_guard<std::mutex> inputLock(tf->inputBGRALock);
					if (tf->inputBGRA.empty()) {
						obs_log(LOG_DEBUG, "Preview image is empty");
						if (tf->source) {
							obs_source_skip_video_filter(tf->source);
						}
						return;
					}
					outputBGRA = tf->inputBGRA;
				} else {
					if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
					    (uint32_t)tf->outputPreviewBGRA.rows != height) {
						if (tf->source) {
							obs_source_skip_video_filter(tf->source);
						}
						return;
					}
					outputBGRA = tf->outputPreviewBGRA;
				}
			}
			outputMask = tf->outputMask;
		}
		const auto render_snapshot_end = std::chrono::steady_clock::now();
		{
			std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
			update_running_avg(
				tf->perfRenderSnapshotMsAvg,
				tf->perfRenderSnapshotSamples,
				std::chrono::duration<double, std::milli>(
					render_snapshot_end - render_snapshot_start)
					.count());
		}

		const auto render_upload_start = std::chrono::steady_clock::now();
		if (tf->uploadTextureWidth != width || tf->uploadTextureHeight != height) {
			if (tf->previewUploadTexture) {
				gs_texture_destroy(tf->previewUploadTexture);
				tf->previewUploadTexture = nullptr;
			}
			if (tf->maskUploadTexture) {
				gs_texture_destroy(tf->maskUploadTexture);
				tf->maskUploadTexture = nullptr;
			}
			tf->uploadTextureWidth = width;
			tf->uploadTextureHeight = height;
		}

		gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
		if (needsPreviewImage) {
			if (!tf->previewUploadTexture) {
				tf->previewUploadTexture =
					gs_texture_create(width, height, GS_BGRA, 1, nullptr, GS_DYNAMIC);
				if (!tf->previewUploadTexture) {
					obs_source_skip_video_filter(tf->source);
					return;
				}
			}

			gs_texture_set_image(tf->previewUploadTexture, outputBGRA.data,
					   (uint32_t)outputBGRA.step, false);
			tex = tf->previewUploadTexture;
		}
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

			if (tf->maskingEnabled && !outputMask.empty()) {
			if (outputMask.empty() || (uint32_t)outputMask.cols != width ||
			    (uint32_t)outputMask.rows != height) {
				obs_source_skip_video_filter(tf->source);
				return;
			}

			if (!tf->maskUploadTexture) {
				tf->maskUploadTexture =
					gs_texture_create(width, height, GS_R8, 1, nullptr, GS_DYNAMIC);
				if (!tf->maskUploadTexture) {
					obs_source_skip_video_filter(tf->source);
					return;
				}
			}

			gs_texture_set_image(tf->maskUploadTexture, outputMask.data,
					   (uint32_t)outputMask.step, false);
			maskTexture = tf->maskUploadTexture;
			gs_effect_set_texture(maskParam, maskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				tex = blur_image(tf, width, height, maskTexture);
				if (!tex) {
					obs_source_skip_video_filter(tf->source);
					return;
				}
			} else if (tf->maskingType == "pixelate") {
				tex = pixelate_image(tf, width, height, maskTexture,
						   (float)tf->maskingBlurRadius);
				if (!tex) {
					obs_source_skip_video_filter(tf->source);
					return;
				}
			} else if (tf->maskingType == "transparent") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, 0);
		} else if (tf->maskingType == "solid_color") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, tf->maskingColor);
		} else if (tf->maskingType == "inpaint") {
				gs_effect_t *inpaintEffect = tf->inpaintEffect;
				gs_eparam_t *iImageParam = gs_effect_get_param_by_name(inpaintEffect, "image");
				gs_eparam_t *iMaskParam = gs_effect_get_param_by_name(inpaintEffect, "focalmask");
				gs_eparam_t *iRadiusParam = gs_effect_get_param_by_name(inpaintEffect, "inpaint_radius");
				gs_eparam_t *iTexSizeParam = gs_effect_get_param_by_name(inpaintEffect, "tex_size");

				gs_effect_set_texture(iImageParam, tex);
				gs_effect_set_texture(iMaskParam, maskTexture);
				if (iRadiusParam) gs_effect_set_float(iRadiusParam, tf->inpaintRadius);
 			if (iTexSizeParam) {
 					gs_effect_set_float(iTexSizeParam, (float)width);
 				}

				const auto inpaint_effect_start = std::chrono::steady_clock::now();

				while (gs_effect_loop(inpaintEffect, "Draw")) {
					gs_draw_sprite(tex, 0, 0, 0);
				}
				const auto inpaint_effect_end = std::chrono::steady_clock::now();
				{
					std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
					update_running_avg(
						tf->perfRenderUploadMsAvg,
						tf->perfRenderUploadSamples,
						std::chrono::duration<double, std::milli>(
							inpaint_effect_start - render_upload_start)
							.count());
					update_running_avg(
						tf->perfRenderEffectMsAvg,
						tf->perfRenderEffectSamples,
						std::chrono::duration<double, std::milli>(
							inpaint_effect_end - inpaint_effect_start)
							.count());
				}
				return;
			}
		}
		const auto render_upload_end = std::chrono::steady_clock::now();
		{
			std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
			update_running_avg(
				tf->perfRenderUploadMsAvg,
				tf->perfRenderUploadSamples,
				std::chrono::duration<double, std::milli>(render_upload_end - render_upload_start)
					.count());
		}

		const auto render_effect_start = std::chrono::steady_clock::now();
		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}
		const auto render_effect_end = std::chrono::steady_clock::now();
		{
			std::lock_guard<std::mutex> perfLock(tf->perfStatsMutex);
			update_running_avg(
				tf->perfRenderEffectMsAvg,
				tf->perfRenderEffectSamples,
				std::chrono::duration<double, std::milli>(render_effect_end - render_effect_start)
					.count());
		}

	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
