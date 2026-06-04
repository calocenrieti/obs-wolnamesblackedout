#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

struct OrtDmlApi;

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ocr {

struct OCRResult {
    std::string text;
    float confidence = 0.0f;
};

class PaddleOCRRecognizer {
public:
    PaddleOCRRecognizer();
    ~PaddleOCRRecognizer();

    void setUseDirectML(bool useDirectML);
    bool loadDictionary(const std::string &dict_file);
    bool loadModel(const std::string &model_path);
    bool isReady() const;

    std::vector<OCRResult> inferBatch(const std::vector<cv::Mat> &images);

private:
    bool initializeDirectML();
    bool prepareSessionIO();
    std::vector<float> preprocessImage(const cv::Mat &image, int targetHeight, int maxWidth) const;
    OCRResult decodeSequence(const float *data, int64_t seqLength, int64_t vocabSize) const;

    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions session_options_;
    Ort::SessionOptions session_options_dml_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    std::vector<std::string> input_node_name_strings_;
    std::vector<std::string> output_node_name_strings_;
    std::vector<const char *> input_node_names_;
    std::vector<const char *> output_node_names_;

    std::vector<std::string> character_list_;
    bool use_directml_ = false;
    bool is_ready_ = false;
};

} // namespace ocr
