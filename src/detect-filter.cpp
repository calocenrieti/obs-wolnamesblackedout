#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
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

#include "yolodetector/YOLODetector.h"

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

		cv::Mat frame = tf->pendingInferenceFrame.clone();
		tf->pendingInferenceFrameReady = false;
		lock.unlock();

		std::vector<Object> objects;
		std::vector<std::string> classNames;

		try {
			std::unique_lock<std::mutex> modelLock(tf->modelMutex);
			if (tf->yolodetector) {
				auto bboxes_opt = tf->yolodetector->inference(frame, tf->conf_threshold);
				if (bboxes_opt.has_value()) {
					objects = tf->yolodetector->convertToObjects(bboxes_opt.value());
					for (const auto &obj : objects) {
						if ((size_t)obj.label >= classNames.size()) {
							classNames.resize(obj.label + 1, "class_" + std::to_string(obj.label));
						}
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

	// preview toggle for the render preview overlay
	// obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								  obs_property_t *,
								  obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_t *prop = obs_properties_get(props_, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");

		obs_property_set_visible(prop, enabled);
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		obs_property_set_visible(masking_dilation, enabled);
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, enabled);
		}
		return true;
	});

 	// add masking options drop down selection: "None", "Solid color", "Blur", "Pixelate", "Transparent", "Inpaint"
 	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
 							       obs_module_text("MaskingType"),
 							       OBS_COMBO_TYPE_LIST,
 							       OBS_COMBO_FORMAT_STRING);
 	obs_property_list_add_string(masking_type, "None", "none");
 	obs_property_list_add_string(masking_type, "Solid color", "solid_color");
 	obs_property_list_add_string(masking_type, "Blur", "blur");
 	obs_property_list_add_string(masking_type, "Pixelate", "pixelate");
 	obs_property_list_add_string(masking_type, "Inpaint", "inpaint");
	obs_property_list_add_string(masking_type, "Transparent", "transparent");


	// add color picker for solid color masking
	obs_properties_add_color(masking_group, "masking_color", obs_module_text("MaskingColor"));

	// add slider for blur radius
	obs_properties_add_int_slider(masking_group, "masking_blur_radius",
				      obs_module_text("MaskingBlurRadius"), 1, 30, 1);

	// add callback to show/hide blur radius and color picker
	obs_property_set_modified_callback(masking_type, [](obs_properties_t *props_,
							    obs_property_t *,
							    obs_data_t *settings) {
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_set_visible(masking_dilation, masking_enabled);

		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, masking_enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, masking_enabled);
		}
		return true;
	});

 	// add slider for dilation iterations
 	obs_properties_add_int_slider(masking_group, "dilation_iterations",
 				      obs_module_text("DilationIterations"), 0, 20, 1);

 	obs_properties_add_float_slider(masking_group, "threshold", obs_module_text("Threshold"), 0.01,
  					1.0, 0.01);

 	// Asynchronous inference toggle
 	obs_properties_add_bool(masking_group, "async_inference", obs_module_text("AsyncInference"));
 	// obs_property_set_description(obs_properties_get(masking_group, "async_inference"),
 		// obs_module_text("AsyncInferenceDescription"));

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

 	// Add a informative text about the plugin
 	std::string basic_info =
 		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
 	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

 	UNUSED_PARAMETER(data);
 	return props;
 }

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", false);
	obs_data_set_default_int(settings, "max_unseen_frames", 10);
	obs_data_set_default_bool(settings, "show_unseen_objects", true);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", false);
	obs_data_set_default_double(settings, "threshold", 0.15);
	obs_data_set_default_string(settings, "model_size", "yolodetector");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_bool(settings, "masking_group", true);
	obs_data_set_default_string(settings, "masking_type", "solid_color");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 0);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_double(settings, "zoom_factor", 0.0);
	obs_data_set_default_double(settings, "zoom_speed_factor", 0.05);
	obs_data_set_default_string(settings, "zoom_object", "single");
	obs_data_set_default_string(settings, "save_detections_path", "");
 	obs_data_set_default_bool(settings, "crop_group", false);
 	obs_data_set_default_int(settings, "crop_left", 0);
 	obs_data_set_default_int(settings, "crop_right", 0);
 	obs_data_set_default_int(settings, "crop_top", 0);
 	obs_data_set_default_int(settings, "crop_bottom", 0);

 	// Exclude range defaults
 	obs_data_set_default_bool(settings, "exclude_group", false);
 	obs_data_set_default_bool(settings, "exclude_preview", true);
 	obs_data_set_default_int(settings, "exclude_left", 0);
 	obs_data_set_default_int(settings, "exclude_right", 0);
 	obs_data_set_default_int(settings, "exclude_top", 0);
 	obs_data_set_default_int(settings, "exclude_bottom", 0);

 	// Inpaint effect defaults
 	obs_data_set_default_double(settings, "inpaint_radius", 70.0);

 	// Asynchronous inference default
 	obs_data_set_default_bool(settings, "async_inference", true);
 }

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "dilation_iterations");

	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
 	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
 	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
 	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");

 	tf->exclude_group_enabled = obs_data_get_bool(settings, "exclude_group");
 	tf->exclude_preview = obs_data_get_bool(settings, "exclude_preview");
 	tf->exclude_left = (int)obs_data_get_int(settings, "exclude_left");
 	tf->exclude_right = (int)obs_data_get_int(settings, "exclude_right");
 	tf->exclude_top = (int)obs_data_get_int(settings, "exclude_top");
 	tf->exclude_bottom = (int)obs_data_get_int(settings, "exclude_bottom");

 	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");

 	// Inpaint parameters
 	tf->inpaintRadius = (float)obs_data_get_double(settings, "inpaint_radius");

 	// Asynchronous inference setting
 	tf->asyncInference = obs_data_get_bool(settings, "async_inference");

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
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
		float nms_th_ = 0.45f;


		// Load model
		try {

			if (tf->modelSize == "yolodetector") {
				// Initialize YOLODetector for yolodetector model size
				if (!tf->yolodetector) {
					tf->yolodetector = std::make_unique<YOLODetector>();
					// GPU 使用設定を適用（DirectML エクスプローラー初期化）
					bool use_gpu = (tf->useGPU == "dml" || tf->useGPU == "cuda");
					tf->yolodetector->setUseGPU(use_gpu);
					
					if (use_gpu) {
						// DirectML 初期化を試行（Windows のみ）
						if (!tf->yolodetector->initializeDirectML()) {
							obs_log(LOG_WARNING, "Failed to initialize DirectML, falling back to CPU");
						}
					}
				}
				if (!tf->yolodetector->loadModel(tf->modelFilepath.c_str())) {
					throw std::runtime_error("Failed to load YOLODetector model");
				}

			}
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());

			return;
		}
	}



	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
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
		obs_log(LOG_INFO, "  Tracking Enabled: %s",
			obs_data_get_bool(settings, "tracking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Zoom Factor: %.2f",
			obs_data_get_double(settings, "zoom_factor"));
		obs_log(LOG_INFO, "  Zoom Object: %s",
			obs_data_get_string(settings, "zoom_object"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	// enable
	tf->isDisabled = false;
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
	tf->inferenceThread = std::thread(inference_thread_proc, tf);

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
		}
		tf->inferenceCv.notify_one();
		if (tf->inferenceThread.joinable()) {
			tf->inferenceThread.join();
		}

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
 		gs_effect_destroy(tf->kawaseBlurEffect);
 		gs_effect_destroy(tf->maskingEffect);
 		gs_effect_destroy(tf->pixelateEffect);
 		gs_effect_destroy(tf->inpaintEffect);
 		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	// Check if either model is available
	if (tf->isDisabled || !tf->yolodetector) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			// No data to process
			return;
		}
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat inferenceFrame;

	cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
	if (tf->crop_enabled) {
		cropRect = cv::Rect(tf->crop_left, tf->crop_top,
				    imageBGRA.cols - tf->crop_left - tf->crop_right,
				    imageBGRA.rows - tf->crop_top - tf->crop_bottom);
		cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
	} else {
		cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
	}

	{
		std::lock_guard<std::mutex> lock(tf->inferenceMutex);
		tf->pendingInferenceFrame = inferenceFrame;
		tf->pendingInferenceFrameReady = true;
		tf->inferenceCompleted = false;
	}
	tf->inferenceCv.notify_one();

	// If synchronous mode, wait for inference to complete
	if (!tf->asyncInference) {
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

	if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
		}
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

	if (tf->minAreaThreshold > 0) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.rect.area() > (float)tf->minAreaThreshold) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
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


	if (!tf->showUnseenObjects) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			objects.end());
	}


	if (tf->preview || tf->maskingEnabled) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		// if (tf->preview && tf->crop_enabled) {
		// 	// draw the crop rectangle on the frame in a dashed line
		// 	drawDashedRectangle(frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
		// }
		// if (tf->preview && objects.size() > 0) {
		// 	draw_objects(frame, objects, classNames);
		// }

		if (tf->preview && tf->exclude_group_enabled && tf->exclude_preview) {
			cv::Rect excludeRect(
				tf->exclude_left,
				tf->exclude_top,
				frame.cols - tf->exclude_left - tf->exclude_right,
				frame.rows - tf->exclude_top - tf->exclude_bottom);
			draw_exclude_preview(frame, excludeRect);
		}
		if (tf->maskingEnabled) {
			cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			for (const Object &obj : objects) {
				// Check if this detection should be excluded from masking
				if (tf->exclude_group_enabled && is_rect_excluded(obj.rect, tf->exclude_left,
									tf->exclude_right, tf->exclude_top, tf->exclude_bottom,
									frame.cols, frame.rows)) {
					continue;  // Skip this detection - don't add to mask
				}
				cv::rectangle(mask, obj.rect, cv::Scalar(255), -1);
			}
			std::lock_guard<std::mutex> lock(tf->outputLock);
			mask.copyTo(tf->outputMask);

			if (tf->maskingDilateIterations > 0) {
				cv::Mat dilatedMask;
				cv::dilate(tf->outputMask, dilatedMask, cv::Mat(),
					   cv::Point(-1, -1), tf->maskingDilateIterations);
				dilatedMask.copyTo(tf->outputMask);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
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
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview is enabled, render the image
	if (tf->preview || tf->maskingEnabled) {
		cv::Mat outputBGRA, outputMask;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
			outputMask = tf->outputMask.clone();
		}

		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
						      (const uint8_t **)&outputBGRA.data, 0);
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		if (tf->maskingEnabled) {
			maskTexture = gs_texture_create(width, height, GS_R8, 1,
							(const uint8_t **)&outputMask.data, 0);
			gs_effect_set_texture(maskParam, maskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				gs_texture_destroy(tex);
				tex = blur_image(tf, width, height, maskTexture);
			} else if (tf->maskingType == "pixelate") {
				gs_texture_destroy(tex);
				tex = pixelate_image(tf, width, height, maskTexture,
						     (float)tf->maskingBlurRadius);
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

				while (gs_effect_loop(inpaintEffect, "Draw")) {
					gs_draw_sprite(tex, 0, 0, 0);
				}
				gs_texture_destroy(tex);
				gs_texture_destroy(maskTexture);
				return;
			}
		}

		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_texture_destroy(tex);
		gs_texture_destroy(maskTexture);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
