#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

struct OrtDmlApi;

#ifdef _WIN32
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <dml_provider_factory.h>
#endif

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <optional>
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
    void setGpuZeroCopyEnabled(bool enabled);
    void setNumThreads(uint32_t numThreads);
    bool loadDictionary(const std::string &dict_file);
    bool loadModel(const std::string &model_path);
    bool isReady() const;

    std::vector<OCRResult> inferBatch(const std::vector<cv::Mat> &images);
    std::vector<OCRResult> inferBatch(const cv::Mat &frameBGRA, const std::vector<cv::Rect> &rects,
                                      int expandPixels = 0);
#ifdef _WIN32
    std::vector<OCRResult> inferBatch(HANDLE sharedHandle, uint32_t frameWidth, uint32_t frameHeight,
                                      const std::vector<cv::Rect> &rects, int expandPixels = 0);
#endif

private:
    bool initializeDirectML();
    bool prepareSessionIO();
    void preprocessImageToBuffer(const cv::Mat &image, int targetHeight, int maxWidth, float *outBuffer) const;
    bool initializeGpuPipeline(int frameWidth, int frameHeight, int batchSize);
    bool ensureGpuInputTexture(const cv::Mat &image);
    std::optional<Ort::Value> runGpuPreprocessBatchToOrtValue(const cv::Mat &frameBGRA,
                                                              const std::vector<cv::Rect> &rects,
                                                              int expandPixels = 0);
#ifdef _WIN32
    std::optional<Ort::Value> runGpuPreprocessBatchToOrtValue(HANDLE sharedHandle,
                                                              uint32_t frameWidth,
                                                              uint32_t frameHeight,
                                                              const std::vector<cv::Rect> &rects,
                                                              int expandPixels = 0);
#endif
    void releaseGpuResources();
    OCRResult decodeSequence(const float *data, int64_t seqLength, int64_t vocabSize) const;

    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions session_options_;
    Ort::SessionOptions session_options_dml_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::unique_ptr<Ort::MemoryInfo> dml_memory_info_;

    std::vector<std::string> input_node_name_strings_;
    std::vector<std::string> output_node_name_strings_;
    std::vector<const char *> input_node_names_;
    std::vector<const char *> output_node_names_;

    std::vector<std::string> character_list_;
    uint32_t num_threads_ = 1;
    bool use_directml_ = false;
    bool gpu_zero_copy_enabled_ = false;
    bool is_ready_ = false;

#ifdef _WIN32
    bool gpu_pipeline_ready_ = false;
    bool gpu_pipeline_initialized_ = false;
    int gpu_batch_size_ = 0;
    int gpu_input_width_ = 0;
    int gpu_input_height_ = 0;
    ID3D11Device *d3d11_device_ = nullptr;
    ID3D11DeviceContext *d3d11_context_ = nullptr;
    ID3D11Texture2D *d3d11_shared_texture_ = nullptr;
    IDXGIKeyedMutex *d3d11_keyed_mutex_ = nullptr;
    HANDLE shared_handle_ = nullptr;
    ID3D12Device *d3d12_device_ = nullptr;
    ID3D12CommandQueue *d3d12_queue_ = nullptr;
    ID3D12CommandAllocator *d3d12_allocator_ = nullptr;
    ID3D12GraphicsCommandList *d3d12_command_list_ = nullptr;
    ID3D12Resource *d3d12_shared_resource_ = nullptr;
    ID3D12Resource *d3d12_uav_buffer_ = nullptr;
    ID3D12Resource *d3d12_constant_buffer_ = nullptr;
    ID3D12Fence *d3d12_fence_ = nullptr;
    HANDLE d3d12_fence_event_ = nullptr;
    UINT64 d3d12_fence_value_ = 0;
    ID3D12RootSignature *d3d12_root_signature_ = nullptr;
    ID3D12PipelineState *d3d12_pipeline_state_ = nullptr;
    ID3D12DescriptorHeap *d3d12_descriptor_heap_ = nullptr;
    void *dml_gpu_allocation_ = nullptr;
#endif
};

} // namespace ocr
