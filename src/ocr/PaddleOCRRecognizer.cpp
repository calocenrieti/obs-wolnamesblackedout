#include "ocr/PaddleOCRRecognizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <dml_provider_factory.h>

#ifdef _WIN32
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <DirectML.h>
#endif

namespace ocr {

#ifdef _WIN32
namespace {
constexpr int kOcrTargetHeight = 48;
constexpr int kOcrMaxWidth = 320;

struct OCRGPUPreprocessConfig {
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t roi_x;
    uint32_t roi_y;
    uint32_t roi_width;
    uint32_t roi_height;
    uint32_t resized_width;
    uint32_t batch_index;
};

static const char *kOCRPreprocessShader = R"(
Texture2D<float4> InputTexture : register(t0);
SamplerState BilinearSampler : register(s0);
RWStructuredBuffer<float> OutputBuffer : register(u0);

cbuffer PreprocessParams : register(b0)
{
    uint FrameWidth;
    uint FrameHeight;
    uint RoiX;
    uint RoiY;
    uint RoiWidth;
    uint RoiHeight;
    uint ResizedWidth;
    uint BatchIndex;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x;
    uint y = dtid.y;
    if (x >= 320 || y >= 48) {
        return;
    }

    float3 pixel = float3(0.0f, 0.0f, 0.0f);
    if (x < ResizedWidth) {
        float u = (RoiX + ((x + 0.5f) * (float)RoiWidth / (float)ResizedWidth)) / (float)FrameWidth;
        float v = (RoiY + ((y + 0.5f) * (float)RoiHeight / 48.0f)) / (float)FrameHeight;
        float4 color = InputTexture.SampleLevel(BilinearSampler, float2(u, v), 0);
        pixel = color.rgb;
    }

    float3 normalized = (pixel - float3(0.5f, 0.5f, 0.5f)) / float3(0.5f, 0.5f, 0.5f);
    uint channelSize = 320 * 48;
    uint batchPlane = channelSize * 3;
    OutputBuffer[BatchIndex * batchPlane + 0 * channelSize + y * 320 + x] = normalized.b;
    OutputBuffer[BatchIndex * batchPlane + 1 * channelSize + y * 320 + x] = normalized.g;
    OutputBuffer[BatchIndex * batchPlane + 2 * channelSize + y * 320 + x] = normalized.r;
}
)";

static cv::Rect expandAndClampRect(const cv::Rect &rect, int expandPixels, int frameWidth, int frameHeight)
{
    cv::Rect expanded = rect;
    if (expandPixels > 0) {
        expanded.x -= expandPixels;
        expanded.y -= expandPixels;
        expanded.width += expandPixels * 2;
        expanded.height += expandPixels * 2;
    }
    expanded &= cv::Rect(0, 0, frameWidth, frameHeight);
    return expanded;
}
} // namespace
#endif

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
#ifdef _WIN32
    dml_memory_info_ = std::make_unique<Ort::MemoryInfo>(
        "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
#endif
}

PaddleOCRRecognizer::~PaddleOCRRecognizer()
{
    releaseGpuResources();
}

void PaddleOCRRecognizer::setUseDirectML(bool useDirectML)
{
    use_directml_ = useDirectML;
}

void PaddleOCRRecognizer::setGpuZeroCopyEnabled(bool enabled)
{
    gpu_zero_copy_enabled_ = enabled;
}

void PaddleOCRRecognizer::setNumThreads(uint32_t numThreads)
{
    num_threads_ = std::max<uint32_t>(1, numThreads);
    session_options_.SetIntraOpNumThreads(static_cast<int>(num_threads_));
    session_options_dml_.SetIntraOpNumThreads(static_cast<int>(num_threads_));
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
        session_options_dml_.SetIntraOpNumThreads(static_cast<int>(num_threads_));

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

#ifdef _WIN32
bool PaddleOCRRecognizer::initializeGpuPipeline(int frameWidth, int frameHeight, int batchSize)
{
    if (gpu_pipeline_ready_ && gpu_batch_size_ == batchSize && gpu_input_width_ == frameWidth && gpu_input_height_ == frameHeight) {
        return true;
    }

    if (gpu_pipeline_initialized_ && !gpu_pipeline_ready_) {
        return false;
    }

    releaseGpuResources();
    gpu_pipeline_initialized_ = true;

    try {
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device_));
        if (FAILED(hr)) {
            return false;
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        hr = d3d12_device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&d3d12_queue_));
        if (FAILED(hr)) {
            return false;
        }

        hr = d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&d3d12_allocator_));
        if (FAILED(hr)) {
            return false;
        }

        hr = d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              d3d12_allocator_, nullptr,
                                              IID_PPV_ARGS(&d3d12_command_list_));
        if (FAILED(hr)) {
            return false;
        }
        d3d12_command_list_->Close();

        hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d12_fence_));
        if (FAILED(hr)) {
            return false;
        }
        d3d12_fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!d3d12_fence_event_) {
            return false;
        }

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.NumDescriptors = 2;
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = d3d12_device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&d3d12_descriptor_heap_));
        if (FAILED(hr)) {
            return false;
        }

        ID3DBlob *shader_blob = nullptr;
        ID3DBlob *error_blob = nullptr;
        hr = D3DCompile(kOCRPreprocessShader, std::strlen(kOCRPreprocessShader), nullptr, nullptr, nullptr,
                        "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shader_blob, &error_blob);
        if (FAILED(hr)) {
            if (error_blob) {
                error_blob->Release();
            }
            return false;
        }

        D3D12_DESCRIPTOR_RANGE descriptor_ranges[2]{};
        descriptor_ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptor_ranges[0].NumDescriptors = 1;
        descriptor_ranges[0].BaseShaderRegister = 0;
        descriptor_ranges[0].RegisterSpace = 0;
        descriptor_ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptor_ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        descriptor_ranges[1].NumDescriptors = 1;
        descriptor_ranges[1].BaseShaderRegister = 0;
        descriptor_ranges[1].RegisterSpace = 0;
        descriptor_ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        D3D12_ROOT_PARAMETER root_parameters[3]{};
        root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        root_parameters[0].Descriptor.ShaderRegister = 0;
        root_parameters[0].Descriptor.RegisterSpace = 0;
        root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
        root_parameters[1].DescriptorTable.pDescriptorRanges = &descriptor_ranges[0];
        root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
        root_parameters[2].DescriptorTable.pDescriptorRanges = &descriptor_ranges[1];
        root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_STATIC_SAMPLER_DESC static_sampler{};
        static_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        static_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        static_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        static_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        static_sampler.MipLODBias = 0.0f;
        static_sampler.MaxAnisotropy = 1;
        static_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        static_sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        static_sampler.MinLOD = 0.0f;
        static_sampler.MaxLOD = D3D12_FLOAT32_MAX;
        static_sampler.ShaderRegister = 0;
        static_sampler.RegisterSpace = 0;
        static_sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC root_sig_desc{};
        root_sig_desc.NumParameters = 3;
        root_sig_desc.pParameters = root_parameters;
        root_sig_desc.NumStaticSamplers = 1;
        root_sig_desc.pStaticSamplers = &static_sampler;
        root_sig_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ID3DBlob *serialized_root_sig = nullptr;
        ID3DBlob *root_sig_error_blob = nullptr;
        hr = D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &serialized_root_sig, &root_sig_error_blob);
        if (SUCCEEDED(hr)) {
            hr = d3d12_device_->CreateRootSignature(0, serialized_root_sig->GetBufferPointer(),
                                                    serialized_root_sig->GetBufferSize(),
                                                    IID_PPV_ARGS(&d3d12_root_signature_));
            serialized_root_sig->Release();
        }
        if (FAILED(hr)) {
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = d3d12_root_signature_;
        pso_desc.CS.pShaderBytecode = shader_blob->GetBufferPointer();
        pso_desc.CS.BytecodeLength = shader_blob->GetBufferSize();
        hr = d3d12_device_->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&d3d12_pipeline_state_));
        shader_blob->Release();
        if (FAILED(hr)) {
            return false;
        }

        D3D12_HEAP_PROPERTIES cb_heap_props{};
        cb_heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC cb_desc{};
        cb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        const UINT64 aligned_cb_size = (sizeof(OCRGPUPreprocessConfig) + 255) & ~255;
        cb_desc.Width = aligned_cb_size * static_cast<UINT64>(batchSize);
        cb_desc.Height = 1;
        cb_desc.DepthOrArraySize = 1;
        cb_desc.MipLevels = 1;
        cb_desc.Format = DXGI_FORMAT_UNKNOWN;
        cb_desc.SampleDesc.Count = 1;
        cb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = d3d12_device_->CreateCommittedResource(&cb_heap_props, D3D12_HEAP_FLAG_NONE, &cb_desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&d3d12_constant_buffer_));
        if (FAILED(hr)) {
            return false;
        }

        const uint32_t output_floats = static_cast<uint32_t>(batchSize) * kOcrTargetHeight * kOcrMaxWidth * 3;
        D3D12_HEAP_PROPERTIES uav_heap_props{};
        uav_heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC uav_desc{};
        uav_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uav_desc.Width = static_cast<UINT64>(output_floats) * sizeof(float);
        uav_desc.Height = 1;
        uav_desc.DepthOrArraySize = 1;
        uav_desc.MipLevels = 1;
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.SampleDesc.Count = 1;
        uav_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        uav_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr = d3d12_device_->CreateCommittedResource(&uav_heap_props, D3D12_HEAP_FLAG_NONE, &uav_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&d3d12_uav_buffer_));
        if (FAILED(hr)) {
            return false;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE heap_start = d3d12_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
        const UINT increment = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE uav_handle = heap_start;
        uav_handle.ptr += increment;

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_view_desc{};
        uav_view_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_view_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_view_desc.Buffer.FirstElement = 0;
        uav_view_desc.Buffer.NumElements = output_floats;
        uav_view_desc.Buffer.StructureByteStride = sizeof(float);
        d3d12_device_->CreateUnorderedAccessView(d3d12_uav_buffer_, nullptr, &uav_view_desc, uav_handle);

        dml_gpu_allocation_ = nullptr;
        const auto &ort_api = Ort::GetApi();
        const OrtDmlApi *dml_api = nullptr;
        OrtStatus *status = ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                            reinterpret_cast<const void **>(&dml_api));
        if (status != nullptr) {
            ort_api.ReleaseStatus(status);
            return false;
        }
        status = dml_api->CreateGPUAllocationFromD3DResource(d3d12_uav_buffer_, &dml_gpu_allocation_);
        if (status != nullptr) {
            ort_api.ReleaseStatus(status);
            return false;
        }

        gpu_batch_size_ = batchSize;
        gpu_input_width_ = frameWidth;
        gpu_input_height_ = frameHeight;
        gpu_pipeline_ready_ = true;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool PaddleOCRRecognizer::ensureGpuInputTexture(const cv::Mat &image)
{
    if (image.empty()) {
        return false;
    }

    if (d3d11_device_ && d3d11_context_ && d3d11_shared_texture_ &&
        gpu_input_width_ == image.cols && gpu_input_height_ == image.rows) {
        return true;
    }

    if (d3d11_shared_texture_) {
        d3d11_shared_texture_->Release();
        d3d11_shared_texture_ = nullptr;
    }
    if (d3d11_keyed_mutex_) {
        d3d11_keyed_mutex_->Release();
        d3d11_keyed_mutex_ = nullptr;
    }
    if (d3d12_shared_resource_) {
        d3d12_shared_resource_->Release();
        d3d12_shared_resource_ = nullptr;
    }
    if (shared_handle_) {
        CloseHandle(shared_handle_);
        shared_handle_ = nullptr;
    }
    if (d3d11_context_) {
        d3d11_context_->Release();
        d3d11_context_ = nullptr;
    }
    if (d3d11_device_) {
        d3d11_device_->Release();
        d3d11_device_ = nullptr;
    }

    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                   D3D11_SDK_VERSION, &d3d11_device_, nullptr, &d3d11_context_);
    if (FAILED(hr) || !d3d11_device_ || !d3d11_context_) {
        return false;
    }

    D3D11_TEXTURE2D_DESC tex_desc{};
    tex_desc.Width = static_cast<UINT>(image.cols);
    tex_desc.Height = static_cast<UINT>(image.rows);
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.SampleDesc.Quality = 0;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    tex_desc.CPUAccessFlags = 0;
    tex_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    hr = d3d11_device_->CreateTexture2D(&tex_desc, nullptr, &d3d11_shared_texture_);
    if (FAILED(hr)) {
        return false;
    }

    hr = d3d11_shared_texture_->QueryInterface(__uuidof(IDXGIKeyedMutex),
                                               reinterpret_cast<void **>(&d3d11_keyed_mutex_));
    if (FAILED(hr)) {
        return false;
    }

    IDXGIResource1 *dxgi_resource = nullptr;
    hr = d3d11_shared_texture_->QueryInterface(__uuidof(IDXGIResource1),
                                               reinterpret_cast<void **>(&dxgi_resource));
    if (FAILED(hr)) {
        return false;
    }
    hr = dxgi_resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                           nullptr, &shared_handle_);
    dxgi_resource->Release();
    if (FAILED(hr)) {
        return false;
    }

    hr = d3d12_device_->OpenSharedHandle(shared_handle_, IID_PPV_ARGS(&d3d12_shared_resource_));
    if (FAILED(hr)) {
        return false;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE heap_start = d3d12_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srv_handle = heap_start;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    d3d12_device_->CreateShaderResourceView(d3d12_shared_resource_, &srv_desc, srv_handle);

    gpu_input_width_ = image.cols;
    gpu_input_height_ = image.rows;
    return true;
}

std::optional<Ort::Value> PaddleOCRRecognizer::runGpuPreprocessBatchToOrtValue(const cv::Mat &frameBGRA,
                                                                              const std::vector<cv::Rect> &rects,
                                                                              int expandPixels)
{
    if (!use_directml_ || !gpu_zero_copy_enabled_ || frameBGRA.empty() || rects.empty() || !dml_memory_info_) {
        return std::nullopt;
    }

    const int batchSize = static_cast<int>(rects.size());
    if (!initializeGpuPipeline(frameBGRA.cols, frameBGRA.rows, batchSize)) {
        return std::nullopt;
    }

    const size_t perImageFloats = static_cast<size_t>(3) * kOcrTargetHeight * kOcrMaxWidth;
    const size_t totalFloats = static_cast<size_t>(batchSize) * perImageFloats;
    const size_t outputBytes = totalFloats * sizeof(float);

    if (!d3d12_constant_buffer_ || !d3d12_command_list_ || !d3d12_descriptor_heap_ ||
        !d3d12_root_signature_ || !d3d12_pipeline_state_ || !d3d12_uav_buffer_ ||
        !d3d12_allocator_ || !d3d12_queue_ || !d3d12_fence_ || !d3d12_fence_event_ ||
        !d3d11_device_ || !d3d11_context_ || !d3d11_keyed_mutex_ || !d3d11_shared_texture_) {
        return std::nullopt;
    }

    if (!ensureGpuInputTexture(frameBGRA)) {
        return std::nullopt;
    }

    HRESULT hr = d3d11_keyed_mutex_->AcquireSync(0, 0);
    if (FAILED(hr)) {
        return std::nullopt;
    }

    D3D11_BOX box{};
    box.left = 0;
    box.top = 0;
    box.front = 0;
    box.right = static_cast<UINT>(frameBGRA.cols);
    box.bottom = static_cast<UINT>(frameBGRA.rows);
    box.back = 1;
    const UINT row_pitch = static_cast<UINT>(frameBGRA.step[0]);
    d3d11_context_->UpdateSubresource(d3d11_shared_texture_, 0, &box, frameBGRA.data, row_pitch, 0);
    d3d11_keyed_mutex_->ReleaseSync(1);

    ID3D12DescriptorHeap *heaps[] = {d3d12_descriptor_heap_};
    D3D12_GPU_DESCRIPTOR_HANDLE descriptor_heap_start = d3d12_descriptor_heap_->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_handle = descriptor_heap_start;
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu_handle = descriptor_heap_start;
    uav_gpu_handle.ptr += d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    void *mapped = nullptr;
    hr = d3d12_constant_buffer_->Map(0, nullptr, &mapped);
    if (FAILED(hr)) {
        return std::nullopt;
    }
    auto *mapped_cb = static_cast<uint8_t *>(mapped);
    const UINT64 aligned_cb_size = (sizeof(OCRGPUPreprocessConfig) + 255) & ~255;

    for (int i = 0; i < batchSize; ++i) {
        cv::Rect rect = expandAndClampRect(rects[static_cast<size_t>(i)], expandPixels, frameBGRA.cols, frameBGRA.rows);
        if (rect.area() <= 0) {
            d3d12_constant_buffer_->Unmap(0, nullptr);
            return std::nullopt;
        }

        const int resizedWidth = std::max(1, std::min(
            kOcrMaxWidth,
            static_cast<int>(std::ceil(static_cast<float>(kOcrTargetHeight) * static_cast<float>(rect.width) /
                                        static_cast<float>(std::max(1, rect.height))))));

        OCRGPUPreprocessConfig cb_data{};
        cb_data.frame_width = static_cast<uint32_t>(frameBGRA.cols);
        cb_data.frame_height = static_cast<uint32_t>(frameBGRA.rows);
        cb_data.roi_x = static_cast<uint32_t>(rect.x);
        cb_data.roi_y = static_cast<uint32_t>(rect.y);
        cb_data.roi_width = static_cast<uint32_t>(rect.width);
        cb_data.roi_height = static_cast<uint32_t>(rect.height);
        cb_data.resized_width = static_cast<uint32_t>(resizedWidth);
        cb_data.batch_index = static_cast<uint32_t>(i);

        std::memcpy(mapped_cb + static_cast<size_t>(i) * static_cast<size_t>(aligned_cb_size), &cb_data, sizeof(cb_data));
    }

    d3d12_constant_buffer_->Unmap(0, nullptr);

    hr = d3d12_allocator_->Reset();
    if (FAILED(hr)) {
        return std::nullopt;
    }
    hr = d3d12_command_list_->Reset(d3d12_allocator_, d3d12_pipeline_state_);
    if (FAILED(hr)) {
        return std::nullopt;
    }

    d3d12_command_list_->SetComputeRootSignature(d3d12_root_signature_);
    d3d12_command_list_->SetPipelineState(d3d12_pipeline_state_);
    d3d12_command_list_->SetDescriptorHeaps(1, heaps);
    d3d12_command_list_->SetComputeRootDescriptorTable(1, srv_gpu_handle);
    d3d12_command_list_->SetComputeRootDescriptorTable(2, uav_gpu_handle);

    const UINT dispatch_x = static_cast<UINT>((kOcrMaxWidth + 15) / 16);
    const UINT dispatch_y = static_cast<UINT>((kOcrTargetHeight + 15) / 16);
    for (int i = 0; i < batchSize; ++i) {
        const D3D12_GPU_VIRTUAL_ADDRESS cb_gpu_address =
            d3d12_constant_buffer_->GetGPUVirtualAddress() + static_cast<UINT64>(i) * aligned_cb_size;
        d3d12_command_list_->SetComputeRootConstantBufferView(0, cb_gpu_address);
        d3d12_command_list_->Dispatch(dispatch_x, dispatch_y, 1);
    }
    d3d12_command_list_->Close();

    ID3D12CommandList *lists[] = {d3d12_command_list_};
    d3d12_queue_->ExecuteCommandLists(1, lists);
    d3d12_fence_value_++;
    d3d12_queue_->Signal(d3d12_fence_, d3d12_fence_value_);
    if (d3d12_fence_->GetCompletedValue() < d3d12_fence_value_) {
        d3d12_fence_->SetEventOnCompletion(d3d12_fence_value_, d3d12_fence_event_);
        WaitForSingleObject(d3d12_fence_event_, INFINITE);
    }

    if (dml_gpu_allocation_ == nullptr) {
        return std::nullopt;
    }

    std::vector<int64_t> input_shape = {batchSize, 3, kOcrTargetHeight, kOcrMaxWidth};
    return Ort::Value::CreateTensor(
        *dml_memory_info_,
        dml_gpu_allocation_,
        outputBytes,
        input_shape.data(),
        input_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

std::optional<Ort::Value> PaddleOCRRecognizer::runGpuPreprocessBatchToOrtValue(HANDLE sharedHandle,
                                                                              uint32_t frameWidth,
                                                                              uint32_t frameHeight,
                                                                              const std::vector<cv::Rect> &rects,
                                                                              int expandPixels)
{
    if (!use_directml_ || !gpu_zero_copy_enabled_ || !sharedHandle || frameWidth == 0 || frameHeight == 0 ||
        rects.empty() || !dml_memory_info_) {
        return std::nullopt;
    }

    const int batchSize = static_cast<int>(rects.size());
    if (!initializeGpuPipeline(static_cast<int>(frameWidth), static_cast<int>(frameHeight), batchSize)) {
        return std::nullopt;
    }

    const size_t perImageFloats = static_cast<size_t>(3) * kOcrTargetHeight * kOcrMaxWidth;
    const size_t totalFloats = static_cast<size_t>(batchSize) * perImageFloats;
    const size_t outputBytes = totalFloats * sizeof(float);

    if (!d3d12_constant_buffer_ || !d3d12_command_list_ || !d3d12_descriptor_heap_ ||
        !d3d12_root_signature_ || !d3d12_pipeline_state_ || !d3d12_uav_buffer_ ||
        !d3d12_allocator_ || !d3d12_queue_ || !d3d12_fence_ || !d3d12_fence_event_ ||
        !d3d12_device_) {
        return std::nullopt;
    }

    ID3D12Resource *source_resource = nullptr;
    HRESULT hr = d3d12_device_->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&source_resource));
    if (FAILED(hr) || !source_resource) {
        return std::nullopt;
    }

    ID3D12DescriptorHeap *heaps[] = {d3d12_descriptor_heap_};
    D3D12_CPU_DESCRIPTOR_HANDLE heap_start_cpu = d3d12_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE heap_start_gpu = d3d12_descriptor_heap_->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srv_handle = heap_start_cpu;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_handle = heap_start_gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu_handle = heap_start_gpu;
    uav_gpu_handle.ptr += d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    d3d12_device_->CreateShaderResourceView(source_resource, &srv_desc, srv_handle);

    void *mapped = nullptr;
    hr = d3d12_constant_buffer_->Map(0, nullptr, &mapped);
    if (FAILED(hr)) {
        source_resource->Release();
        return std::nullopt;
    }
    auto *mapped_cb = static_cast<uint8_t *>(mapped);
    const UINT64 aligned_cb_size = (sizeof(OCRGPUPreprocessConfig) + 255) & ~255;

    for (int i = 0; i < batchSize; ++i) {
        cv::Rect rect = expandAndClampRect(rects[static_cast<size_t>(i)], expandPixels,
                                           static_cast<int>(frameWidth), static_cast<int>(frameHeight));
        if (rect.area() <= 0) {
            d3d12_constant_buffer_->Unmap(0, nullptr);
            source_resource->Release();
            return std::nullopt;
        }

        const int resizedWidth = std::max(1, std::min(
            kOcrMaxWidth,
            static_cast<int>(std::ceil(static_cast<float>(kOcrTargetHeight) * static_cast<float>(rect.width) /
                                        static_cast<float>(std::max(1, rect.height))))));

        OCRGPUPreprocessConfig cb_data{};
        cb_data.frame_width = frameWidth;
        cb_data.frame_height = frameHeight;
        cb_data.roi_x = static_cast<uint32_t>(rect.x);
        cb_data.roi_y = static_cast<uint32_t>(rect.y);
        cb_data.roi_width = static_cast<uint32_t>(rect.width);
        cb_data.roi_height = static_cast<uint32_t>(rect.height);
        cb_data.resized_width = static_cast<uint32_t>(resizedWidth);
        cb_data.batch_index = static_cast<uint32_t>(i);

        std::memcpy(mapped_cb + static_cast<size_t>(i) * static_cast<size_t>(aligned_cb_size), &cb_data, sizeof(cb_data));
    }

    d3d12_constant_buffer_->Unmap(0, nullptr);

    hr = d3d12_allocator_->Reset();
    if (FAILED(hr)) {
        source_resource->Release();
        return std::nullopt;
    }
    hr = d3d12_command_list_->Reset(d3d12_allocator_, d3d12_pipeline_state_);
    if (FAILED(hr)) {
        source_resource->Release();
        return std::nullopt;
    }

    d3d12_command_list_->SetComputeRootSignature(d3d12_root_signature_);
    d3d12_command_list_->SetPipelineState(d3d12_pipeline_state_);
    d3d12_command_list_->SetDescriptorHeaps(1, heaps);
    d3d12_command_list_->SetComputeRootConstantBufferView(0, d3d12_constant_buffer_->GetGPUVirtualAddress());
    d3d12_command_list_->SetComputeRootDescriptorTable(1, srv_gpu_handle);
    d3d12_command_list_->SetComputeRootDescriptorTable(2, uav_gpu_handle);

    const UINT dispatch_x = static_cast<UINT>((kOcrMaxWidth + 15) / 16);
    const UINT dispatch_y = static_cast<UINT>((kOcrTargetHeight + 15) / 16);
    d3d12_command_list_->Dispatch(dispatch_x, dispatch_y, 1);
    d3d12_command_list_->Close();

    ID3D12CommandList *lists[] = {d3d12_command_list_};
    d3d12_queue_->ExecuteCommandLists(1, lists);
    d3d12_fence_value_++;
    d3d12_queue_->Signal(d3d12_fence_, d3d12_fence_value_);
    if (d3d12_fence_->GetCompletedValue() < d3d12_fence_value_) {
        d3d12_fence_->SetEventOnCompletion(d3d12_fence_value_, d3d12_fence_event_);
        WaitForSingleObject(d3d12_fence_event_, INFINITE);
    }

    source_resource->Release();

    if (dml_gpu_allocation_ == nullptr) {
        return std::nullopt;
    }

    std::vector<int64_t> input_shape = {batchSize, 3, kOcrTargetHeight, kOcrMaxWidth};
    return Ort::Value::CreateTensor(
        *dml_memory_info_,
        dml_gpu_allocation_,
        outputBytes,
        input_shape.data(),
        input_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

void PaddleOCRRecognizer::releaseGpuResources()
{
    if (dml_gpu_allocation_) {
        const auto &api = Ort::GetApi();
        const OrtDmlApi *dmlApi = nullptr;
        OrtStatus *status = api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&dmlApi));
        if (status == nullptr && dmlApi != nullptr) {
            dmlApi->FreeGPUAllocation(dml_gpu_allocation_);
        }
        dml_gpu_allocation_ = nullptr;
    }
    if (d3d12_fence_event_) {
        CloseHandle(d3d12_fence_event_);
        d3d12_fence_event_ = nullptr;
    }
    if (d3d11_context_) {
        d3d11_context_->Release();
        d3d11_context_ = nullptr;
    }
    if (d3d11_device_) {
        d3d11_device_->Release();
        d3d11_device_ = nullptr;
    }
    if (d3d11_keyed_mutex_) {
        d3d11_keyed_mutex_->Release();
        d3d11_keyed_mutex_ = nullptr;
    }
    if (d3d11_shared_texture_) {
        d3d11_shared_texture_->Release();
        d3d11_shared_texture_ = nullptr;
    }
    if (d3d12_command_list_) {
        d3d12_command_list_->Release();
        d3d12_command_list_ = nullptr;
    }
    if (d3d12_allocator_) {
        d3d12_allocator_->Release();
        d3d12_allocator_ = nullptr;
    }
    if (d3d12_queue_) {
        d3d12_queue_->Release();
        d3d12_queue_ = nullptr;
    }
    if (d3d12_shared_resource_) {
        d3d12_shared_resource_->Release();
        d3d12_shared_resource_ = nullptr;
    }
    if (d3d12_uav_buffer_) {
        d3d12_uav_buffer_->Release();
        d3d12_uav_buffer_ = nullptr;
    }
    if (d3d12_constant_buffer_) {
        d3d12_constant_buffer_->Release();
        d3d12_constant_buffer_ = nullptr;
    }
    if (d3d12_descriptor_heap_) {
        d3d12_descriptor_heap_->Release();
        d3d12_descriptor_heap_ = nullptr;
    }
    if (d3d12_root_signature_) {
        d3d12_root_signature_->Release();
        d3d12_root_signature_ = nullptr;
    }
    if (d3d12_pipeline_state_) {
        d3d12_pipeline_state_->Release();
        d3d12_pipeline_state_ = nullptr;
    }
    if (d3d12_fence_) {
        d3d12_fence_->Release();
        d3d12_fence_ = nullptr;
    }
    if (d3d12_device_) {
        d3d12_device_->Release();
        d3d12_device_ = nullptr;
    }
    if (shared_handle_) {
        CloseHandle(shared_handle_);
        shared_handle_ = nullptr;
    }
    gpu_pipeline_ready_ = false;
    gpu_pipeline_initialized_ = false;
    gpu_batch_size_ = 0;
    gpu_input_width_ = 0;
    gpu_input_height_ = 0;
}
#endif

void PaddleOCRRecognizer::preprocessImageToBuffer(const cv::Mat &image,
                                                  int targetHeight,
                                                  int maxWidth,
                                                  float *outBuffer) const
{
    cv::Mat roi;
    if (image.channels() == 4) {
        cv::cvtColor(image, roi, cv::COLOR_BGRA2BGR);
    } else {
        roi = image;
    }

    if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) {
        std::fill(outBuffer, outBuffer + static_cast<size_t>(3) * targetHeight * maxWidth, 0.0f);
        return;
    }

    const float ratio = static_cast<float>(roi.cols) / static_cast<float>(roi.rows);
    int resizedWidth = static_cast<int>(std::ceil(targetHeight * ratio));
    if (resizedWidth > maxWidth) {
        resizedWidth = maxWidth;
    }

    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(resizedWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    cv::Mat resizedFloat;
    resized.convertTo(resizedFloat, CV_32FC3, 1.0f / 127.5f, -1.0f);

    cv::Mat padded(targetHeight, maxWidth, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    resizedFloat.copyTo(padded(cv::Rect(0, 0, resizedFloat.cols, resizedFloat.rows)));

    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0f, cv::Size(maxWidth, targetHeight), cv::Scalar(0, 0, 0), false, false, CV_32F);

    const size_t blobSize = blob.total();
    if (blob.isContinuous()) {
        std::memcpy(outBuffer, blob.ptr<float>(), blobSize * sizeof(float));
    } else {
        cv::Mat contiguous = blob.clone();
        std::memcpy(outBuffer, contiguous.ptr<float>(), blobSize * sizeof(float));
    }
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

    const size_t tensorSize = static_cast<size_t>(batchSize) * channels * targetHeight * maxWidth;
    std::vector<float> inputTensor(tensorSize);
    const size_t imageTensorSize = static_cast<size_t>(channels) * targetHeight * maxWidth;
    for (int i = 0; i < batchSize; ++i) {
        preprocessImageToBuffer(images[static_cast<size_t>(i)],
                                targetHeight,
                                maxWidth,
                                inputTensor.data() + static_cast<size_t>(i) * imageTensorSize);
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

std::vector<OCRResult> PaddleOCRRecognizer::inferBatch(const cv::Mat &frameBGRA,
                                                      const std::vector<cv::Rect> &rects,
                                                      int expandPixels)
{
    std::vector<OCRResult> results;
    if (!isReady() || frameBGRA.empty() || rects.empty()) {
        return results;
    }

#ifdef _WIN32
    if (use_directml_ && gpu_zero_copy_enabled_) {
        auto gpu_input_tensor = runGpuPreprocessBatchToOrtValue(frameBGRA, rects, expandPixels);
        if (gpu_input_tensor.has_value()) {
            auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                               input_node_names_.data(),
                                               &*gpu_input_tensor,
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
    }
#endif

    const int targetHeight = 48;
    const int maxWidth = 320;
    const int channels = 3;
    const size_t tensorSize = static_cast<size_t>(rects.size()) * channels * targetHeight * maxWidth;
    std::vector<float> inputTensor(tensorSize);
    const size_t imageTensorSize = static_cast<size_t>(channels) * targetHeight * maxWidth;

    for (size_t i = 0; i < rects.size(); ++i) {
        cv::Rect rect = expandAndClampRect(rects[i], expandPixels, frameBGRA.cols, frameBGRA.rows);
        if (rect.area() <= 0) {
            continue;
        }
        cv::Mat roi = frameBGRA(rect);
        preprocessImageToBuffer(roi, targetHeight, maxWidth, inputTensor.data() + i * imageTensorSize);
    }

    std::vector<int64_t> input_shape = {static_cast<int64_t>(rects.size()), channels, targetHeight, maxWidth};
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

#ifdef _WIN32
std::vector<OCRResult> PaddleOCRRecognizer::inferBatch(HANDLE sharedHandle,
                                                      uint32_t frameWidth,
                                                      uint32_t frameHeight,
                                                      const std::vector<cv::Rect> &rects,
                                                      int expandPixels)
{
    std::vector<OCRResult> results;
    if (!isReady() || !sharedHandle || frameWidth == 0 || frameHeight == 0 || rects.empty()) {
        return results;
    }

    if (use_directml_ && gpu_zero_copy_enabled_) {
        auto gpu_input_tensor = runGpuPreprocessBatchToOrtValue(sharedHandle, frameWidth, frameHeight, rects, expandPixels);
        if (gpu_input_tensor.has_value()) {
            auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                               input_node_names_.data(),
                                               &*gpu_input_tensor,
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
    }

    return results;
}
#endif

} // namespace ocr
