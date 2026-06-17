#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>

#include "yolodetector/YOLODetector.h"
#include "ByteTrack/Detection.h"
#include "ByteTrack/Track.h"
#include "ByteTrack/BYTETracker.h"
#include "ocr/PaddleOCRRecognizer.h"

#include <chrono>
#include <unordered_map>

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

	int minAreaThreshold;
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

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::mutex inferenceMutex;
	std::condition_variable inferenceCv;
	cv::Mat pendingInferenceFrame;
	bool pendingInferenceFrameReady = false;
	bool stopInferenceThread = false;
	std::thread inferenceThread;

	std::mutex ocrMutex;
	std::condition_variable ocrCv;
	bool pendingOcrWork = false;
	bool stopOcrThread = false;
	std::vector<cv::Mat> pendingOcrImages;
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
	double ocrRefreshInterval = 3.0;
	int ocrExpandPixels = 0;
	double ocrInitialThreshold = 0.8;
	double ocrContinueThreshold = 0.7;
	std::chrono::steady_clock::time_point lastOcrRefreshTime = std::chrono::steady_clock::now();

	// Asynchronous inference toggle
	bool asyncInference = true;
	bool inferenceCompleted = false;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
