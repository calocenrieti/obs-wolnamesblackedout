#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>

#ifdef _WIN32
#include <d3d11.h>
#endif

#include "yolodetector/YOLODetector.h"
#include "ByteTrack/Detection.h"
#include "ByteTrack/Track.h"
#include "ByteTrack/BYTETracker.h"
#include "ocr/PaddleOCRRecognizer.h"

#include <chrono>
#include <cstdint>
#include <unordered_map>

struct PendingInferenceFrame {
	cv::Mat frameBGRA;
	cv::Mat frameBGR;
	HANDLE gpuSharedHandle = nullptr;
	uint32_t gpuWidth = 0;
	uint32_t gpuHeight = 0;

	bool empty() const
	{
		return frameBGRA.empty() && frameBGR.empty() && gpuSharedHandle == nullptr;
	}

	void clear()
	{
		frameBGRA.release();
		frameBGR.release();
		if (gpuSharedHandle) {
			CloseHandle(gpuSharedHandle);
			gpuSharedHandle = nullptr;
		}
		gpuWidth = 0;
		gpuHeight = 0;
	}

	bool hasBGRA() const
	{
		return !frameBGRA.empty();
	}

	bool hasBGR() const
	{
		return !frameBGR.empty();
	}

	bool hasGpuSharedHandle() const
	{
		return gpuSharedHandle != nullptr;
	}

	const cv::Mat &getBGR() const
	{
		return hasBGR() ? frameBGR : frameBGRA;
	}
};

/**
  * @brief The filter_data struct
  *
  * This struct is used to store the base data needed for ORT filters.
  *
*/
struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;
	int objectCategory;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
 	int maskingDilateIterations;
	std::string maskExcludeText;
	// parsed list of exclude texts (comma-separated in settings)
	std::vector<std::string> maskExcludeTexts;

   	// Inpaint parameters
   	float inpaintRadius;

 	bool trackingEnabled;

	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;

	// Exclude range parameters (detection exclusion area)
	bool exclude_group_enabled;
	bool exclude_preview;
	int exclude_left;
	int exclude_right;
	int exclude_top;
	int exclude_bottom;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;
	gs_effect_t *pixelateEffect;
	gs_effect_t *inpaintEffect;
	gs_texture_t *previewUploadTexture = nullptr;
	gs_texture_t *maskUploadTexture = nullptr;
	gs_texture_t *effectWorkTexture = nullptr;
#ifdef _WIN32
	ID3D11Texture2D *gpuInputTexture = nullptr;
	HANDLE gpuInputSharedHandle = nullptr;
#endif
	uint32_t gpuInputTextureWidth = 0;
	uint32_t gpuInputTextureHeight = 0;
	uint32_t uploadTextureWidth = 0;
	uint32_t uploadTextureHeight = 0;
	uint32_t effectWorkTextureWidth = 0;
	uint32_t effectWorkTextureHeight = 0;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex gpuFrameLock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::mutex inferenceMutex;
	std::condition_variable inferenceCv;
	PendingInferenceFrame pendingInferenceFrame;
	bool pendingInferenceFrameReady = false;
	bool stopInferenceThread = false;
	std::thread inferenceThread;

	std::mutex ocrMutex;
	std::condition_variable ocrCv;
	bool pendingOcrWork = false;
	bool stopOcrThread = false;
	cv::Mat pendingOcrFrameBGRA;
	HANDLE pendingOcrGpuSharedHandle = nullptr;
	uint32_t pendingOcrGpuWidth = 0;
	uint32_t pendingOcrGpuHeight = 0;
	std::vector<cv::Rect> pendingOcrRects;
	std::vector<uint64_t> pendingOcrTrackIds;
	std::thread ocrThread;

	std::vector<Object> latestInferenceObjects;
	std::unordered_map<uint64_t, std::string> latestOcrTexts;
	std::mutex latestObjectsLock;

	std::unique_ptr<YOLODetector> yolodetector;
	std::vector<std::string> classNames;

	// ByteTrack tracker instance (initialized when tracking is enabled)
	std::unique_ptr<byte_track::BYTETracker<ByteTrack::Detection, ByteTrack::Track>> tracker;

	std::unique_ptr<ocr::PaddleOCRRecognizer> ocrRecognizer;
	std::string ocrModelFilepath;
	std::string ocrDictFilepath;
	bool ocrEnabled = false;
	int ocrExpandPixels = 0;
	uint32_t ocrMaxRoisPerFrame = 6;
	double ocrInitialThreshold = 0.8;
	double ocrContinueThreshold = 0.7;
	std::chrono::steady_clock::time_point lastOcrRefreshTime = std::chrono::steady_clock::now();

	// Asynchronous inference toggle
	bool asyncInference = true;
	uint32_t inferenceIntervalFrames = 1;
	uint32_t inferenceIntervalCounter = 0;
	bool inferenceCompleted = false;
	bool gpuZeroCopyEnabled = false;
	HANDLE latestGpuSharedHandle = nullptr;
	uint32_t latestGpuWidth = 0;
	uint32_t latestGpuHeight = 0;

	// Lightweight performance logging (moving averages)
	bool perfLogEnabled = false;
	uint32_t perfLogInterval = 120;
	double perfTickMsAvg = 0.0;
	double perfYoloMsAvg = 0.0;
	double perfOcrMsAvg = 0.0;
	double perfInputCopyMsAvg = 0.0;
	double perfColorConvertMsAvg = 0.0;
	double perfMaskBuildMsAvg = 0.0;
	double perfPublishMsAvg = 0.0;
	double perfRenderCaptureMsAvg = 0.0;
	double perfRenderSnapshotMsAvg = 0.0;
	double perfRenderUploadMsAvg = 0.0;
	double perfRenderEffectMsAvg = 0.0;
	double perfOcrRoiCountAvg = 0.0;
	uint64_t perfRenderTextureCaptureCount = 0;
	uint64_t perfStageSurfaceFallbackCount = 0;
	uint64_t perfTickSamples = 0;
	uint64_t perfYoloSamples = 0;
	uint64_t perfOcrSamples = 0;
	uint64_t perfOcrRoiSamples = 0;
	uint64_t perfInputCopySamples = 0;
	uint64_t perfColorConvertSamples = 0;
	uint64_t perfMaskBuildSamples = 0;
	uint64_t perfPublishSamples = 0;
	uint64_t perfRenderCaptureSamples = 0;
	uint64_t perfRenderSnapshotSamples = 0;
	uint64_t perfRenderUploadSamples = 0;
	uint64_t perfRenderEffectSamples = 0;
	uint32_t perfLogCounter = 0;
	std::mutex perfStatsMutex;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
