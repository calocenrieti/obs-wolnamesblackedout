#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>

#include "yolodetector/YOLODetector.h"

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

   	// Inpaint parameters
   	float inpaintRadius;

 	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	std::string saveDetectionsPath;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// Exclude range parameters (detection exclusion area)
	bool exclude_group_enabled;
	bool exclude_preview;
	int exclude_left;
	int exclude_right;
	int exclude_top;
	int exclude_bottom;

	// // create SORT tracker
	// Sort tracker;

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

	std::vector<Object> latestInferenceObjects;
	std::mutex latestObjectsLock;

	std::unique_ptr<YOLODetector> yolodetector;
	std::vector<std::string> classNames;

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
