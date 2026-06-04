#include "ocr/PaddleOCRRecognizer.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <dml_provider_factory.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace ocr {

PaddleOCRRecognizer::PaddleOCRRecognizer()
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PaddleOCRRecognizer")),
      session_options_(),
      session_options_dml_(),
      session_(nullptr),
      use_directml_(false),
      is_ready_(false)
{
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetLogSeverityLevel(3);
    session_options_.SetIntraOpNumThreads(1);
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault));
}

PaddleOCRRecognizer::~PaddleOCRRecognizer() = default;

void PaddleOCRRecognizer::setUseDirectML(bool useDirectML)
{
    use_directml_ = useDirectML;
}

bool PaddleOCRRecognizer::loadDictionary(const std::string &dict_file)
{
    character_list_.clear();
    character_list_.push_back("blank");

    std::ifstream ifs(dict_file);
    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            character_list_.push_back(line);
        }
    }

    if (std::find(character_list_.begin(), character_list_.end(), " ") == character_list_.end()) {
        character_list_.push_back(" ");
    }

    return true;
}

bool PaddleOCRRecognizer::initializeDirectML()
{
#ifdef _WIN32
    try {
        session_options_dml_ = Ort::SessionOptions();
        session_options_dml_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_dml_.SetLogSeverityLevel(3);
        session_options_dml_.SetIntraOpNumThreads(1);

        const auto &api = Ort::GetApi();
        OrtDmlApi *dmlApi = nullptr;
        Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dmlApi));
        if (dmlApi == nullptr) {
            return false;
        }

        Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
            session_options_dml_, 0));

        return true;
    } catch (const std::exception &) {
        return false;
    }
#else
    return false;
#endif
}

bool PaddleOCRRecognizer::loadModel(const std::string &model_path)
{
    if (character_list_.empty()) {
        return false;
    }

    try {
        if (use_directml_) {
#ifdef _WIN32
            if (!initializeDirectML()) {
                std::wstring model_path_w;
                int outLength = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
                if (outLength == 0) {
                    return false;
                }
                model_path_w.resize(outLength);
                MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, model_path_w.data(), outLength);
                session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options_);
            } else {
                std::wstring model_path_w;
                int outLength = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
                if (outLength == 0) {
                    return false;
                }
                model_path_w.resize(outLength);
                MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, model_path_w.data(), outLength);
                session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options_dml_);
            }
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);
#endif
        } else {
#ifdef _WIN32
            std::wstring model_path_w;
            int outLength = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
            if (outLength == 0) {
                return false;
            }
            model_path_w.resize(outLength);
            MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, model_path_w.data(), outLength);
            session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options_);
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);
#endif
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t num_input_nodes = session_->GetInputCount();
        const size_t num_output_nodes = session_->GetOutputCount();

        input_node_name_strings_.clear();
        output_node_name_strings_.clear();
        input_node_names_.clear();
        output_node_names_.clear();

        input_node_name_strings_.reserve(num_input_nodes);
        output_node_name_strings_.reserve(num_output_nodes);
        input_node_names_.reserve(num_input_nodes);
        output_node_names_.reserve(num_output_nodes);

        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_node_name_strings_.push_back(input_name.get());
            input_node_names_.push_back(input_node_name_strings_.back().c_str());
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_node_name_strings_.push_back(output_name.get());
            output_node_names_.push_back(output_node_name_strings_.back().c_str());
        }

        is_ready_ = true;
        return true;
    } catch (const Ort::Exception &) {
        is_ready_ = false;
        return false;
    }
}

bool PaddleOCRRecognizer::isReady() const
{
    return is_ready_ && session_ != nullptr;
}

std::vector<float> PaddleOCRRecognizer::preprocessImage(const cv::Mat &image, int targetHeight, int maxWidth) const
{
    cv::Mat roi = image;
    const float ratio = static_cast<float>(roi.cols) / static_cast<float>(roi.rows);
    int resizedWidth = static_cast<int>(std::ceil(targetHeight * ratio));
    if (resizedWidth > maxWidth) {
        resizedWidth = maxWidth;
    }

    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(resizedWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 127.5f, -1.0f);

    cv::Mat padded(targetHeight, maxWidth, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    floatImg.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

    std::vector<float> result;
    result.reserve(maxWidth * targetHeight * 3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < maxWidth; x++) {
                result.push_back(padded.at<cv::Vec3f>(y, x)[c]);
            }
        }
    }
    return result;
}

OCRResult PaddleOCRRecognizer::decodeSequence(const float *data, int64_t seqLength, int64_t vocabSize) const
{
    OCRResult result;
    int lastIndex = -1;
    float sumConfidence = 0.0f;
    int tokenCount = 0;

    for (int64_t t = 0; t < seqLength; t++) {
        const float *row = data + t * vocabSize;
        int maxIndex = 0;
        float maxScore = row[0];
        for (int64_t v = 1; v < vocabSize; v++) {
            if (row[v] > maxScore) {
                maxScore = row[v];
                maxIndex = static_cast<int>(v);
            }
        }

        if (maxIndex == 0) {
            lastIndex = 0;
            continue;
        }
        if (maxIndex == lastIndex) {
            continue;
        }

        if (maxIndex < static_cast<int>(character_list_.size())) {
            result.text += character_list_[maxIndex];
            sumConfidence += maxScore;
            tokenCount++;
        }
        lastIndex = maxIndex;
    }

    if (tokenCount > 0) {
        result.confidence = sumConfidence / static_cast<float>(tokenCount);
    }
    return result;
}

std::vector<OCRResult> PaddleOCRRecognizer::inferBatch(const std::vector<cv::Mat> &images)
{
    std::vector<OCRResult> results;
    if (!isReady() || images.empty()) {
        return results;
    }

    const int targetHeight = 48;
    const int maxWidth = 320;
    const int channels = 3;
    const int batchSize = static_cast<int>(images.size());

    std::vector<std::vector<float>> preprocessed;
    preprocessed.reserve(batchSize);
    for (const auto &image : images) {
        preprocessed.push_back(preprocessImage(image, targetHeight, maxWidth));
    }

    const size_t tensorSize = static_cast<size_t>(batchSize) * channels * targetHeight * maxWidth;
    std::vector<float> inputTensor;
    inputTensor.reserve(tensorSize);
    for (const auto &buffer : preprocessed) {
        inputTensor.insert(inputTensor.end(), buffer.begin(), buffer.end());
    }

    std::vector<int64_t> input_shape = {batchSize, channels, targetHeight, maxWidth};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, inputTensor.data(), inputTensor.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_node_names_.data(),
                                       &input_tensor,
                                       1,
                                       output_node_names_.data(),
                                       output_node_names_.size());

    Ort::Value &output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();
    if (output_shape.size() != 3) {
        return results;
    }

    int64_t batch = output_shape[0];
    int64_t seqLength = output_shape[1];
    int64_t vocabSize = output_shape[2];
    const float *output_data = output_tensor.GetTensorData<float>();

    results.reserve(static_cast<size_t>(batch));
    for (int64_t i = 0; i < batch; i++) {
        const float *sequence_data = output_data + i * seqLength * vocabSize;
        results.push_back(decodeSequence(sequence_data, seqLength, vocabSize));
    }

    return results;
}

} // namespace ocr
