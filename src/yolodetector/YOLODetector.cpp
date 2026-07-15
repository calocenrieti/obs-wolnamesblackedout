#include "yolodetector/YOLODetector.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <plugin-support.h>

#ifdef _WIN32
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <DirectML.h>
#include <dml_provider_factory.h>
#endif // _WIN32

// Private 構造体の定義
struct YOLODetector::Private {
    std::unique_ptr<Ort::Env> env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;

    size_t num_input_nodes = 0;
    size_t num_output_nodes = 0;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<std::string> input_node_name_strings;
    std::vector<std::string> output_node_name_strings;

    bool use_directml = false;
    bool gpu_zero_copy_enabled = false;
    uint32_t num_threads = 1;

    cv::Mat resized_img;
    cv::Mat padded_img;
    std::vector<float> input_tensor_buffer;
    
    // DirectML 用
#ifdef _WIN32
    std::unique_ptr<Ort::MemoryInfo> dml_memory_info;
    Ort::SessionOptions session_options_dml;
    bool gpu_pipeline_ready = false;
    bool gpu_pipeline_initialized = false;
    int gpu_width = 0;
    int gpu_height = 0;
    ID3D11Device* d3d11_device = nullptr;
    ID3D11DeviceContext* d3d11_context = nullptr;
    ID3D11Texture2D* d3d11_shared_texture = nullptr;
    IDXGIKeyedMutex* d3d11_keyed_mutex = nullptr;
    HANDLE shared_handle = nullptr;
    ID3D12Device* d3d12_device = nullptr;
    ID3D12CommandQueue* d3d12_queue = nullptr;
    ID3D12CommandAllocator* d3d12_allocator = nullptr;
    ID3D12GraphicsCommandList* d3d12_command_list = nullptr;
    ID3D12Resource* d3d12_shared_resource = nullptr;
    ID3D12Resource* d3d12_upload_buffer = nullptr;
    ID3D12Resource* d3d12_uav_buffer = nullptr;
    ID3D12Resource* d3d12_constant_buffer = nullptr;
    void* d3d12_upload_mapped_ptr = nullptr;
    void* d3d12_constant_mapped_ptr = nullptr;
    ID3D12Fence* d3d12_fence = nullptr;
    HANDLE d3d12_fence_event = nullptr;
    UINT64 d3d12_fence_value = 0;
    ID3D12RootSignature* d3d12_root_signature = nullptr;
    ID3D12PipelineState* d3d12_pipeline_state = nullptr;
    ID3D12DescriptorHeap* d3d12_descriptor_heap = nullptr;
    void* dml_gpu_allocation = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT d3d12_upload_footprint{};
#endif
};

// コンストラクタ
YOLODetector::YOLODetector()
    : m_(std::make_unique<Private>())
{
    resizeScales = 1.0f;
    try {
        m_->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
                                       OrtMemType::OrtMemTypeDefault));

        // ONNX Runtime 初期化（CPU のみサポート）
        m_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");
        
        m_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_->session_options.SetLogSeverityLevel(3);
        m_->session_options.SetIntraOpNumThreads(static_cast<int>(m_->num_threads));
#ifdef _WIN32
        m_->dml_memory_info = std::make_unique<Ort::MemoryInfo>(
            "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
#endif
    }
    catch (const std::exception& e) {
        // 初期化エラーは内部で吸収
    }
}

// デストラクタ
YOLODetector::~YOLODetector() = default;

#ifdef _WIN32
namespace {
struct GPUPreprocessConfig {
    uint32_t src_width;
    uint32_t src_height;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t unpad_width;
    uint32_t unpad_height;
    uint32_t pad_x;
    uint32_t pad_y;
    float mean_r;
    float mean_g;
    float mean_b;
    float padding2;
    float std_r;
    float std_g;
    float std_b;
    float padding3;
};

static const char* kPreprocessShader = R"(
Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);

cbuffer PreprocessParams : register(b0)
{
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;
    uint UnpadWidth;
    uint UnpadHeight;
    uint PadX;
    uint PadY;
    float3 Mean;
    float Dummy1;
    float3 Std;
    float Dummy2;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x;
    uint y = dtid.y;
    if (x >= DstWidth || y >= DstHeight) {
        return;
    }

    float3 pixel = float3(114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f);
    if (x >= PadX && y >= PadY && x < PadX + UnpadWidth && y < PadY + UnpadHeight) {
        float src_x = (((float)(x - PadX) + 0.5f) * (float)SrcWidth / (float)UnpadWidth) - 0.5f;
        float src_y = (((float)(y - PadY) + 0.5f) * (float)SrcHeight / (float)UnpadHeight) - 0.5f;
        int x0 = (int)floor(src_x);
        int y0 = (int)floor(src_y);
        int x1 = min(x0 + 1, (int)SrcWidth - 1);
        int y1 = min(y0 + 1, (int)SrcHeight - 1);
        int ix0 = clamp(x0, 0, (int)SrcWidth - 1);
        int iy0 = clamp(y0, 0, (int)SrcHeight - 1);
        int ix1 = clamp(x1, 0, (int)SrcWidth - 1);
        int iy1 = clamp(y1, 0, (int)SrcHeight - 1);
        float fx = src_x - (float)x0;
        float fy = src_y - (float)y0;
        float4 c00 = InputTexture.Load(int3(ix0, iy0, 0));
        float4 c10 = InputTexture.Load(int3(ix1, iy0, 0));
        float4 c01 = InputTexture.Load(int3(ix0, iy1, 0));
        float4 c11 = InputTexture.Load(int3(ix1, iy1, 0));
        float4 top = lerp(c00, c10, fx);
        float4 bottom = lerp(c01, c11, fx);
        pixel = lerp(top, bottom, fy).rgb;
    }

    float3 normalized = (pixel - Mean) / Std;
    uint channelSize = DstWidth * DstHeight;
    uint baseIndex = y * DstWidth + x;
    OutputBuffer[0 * channelSize + baseIndex] = normalized.r;
    OutputBuffer[1 * channelSize + baseIndex] = normalized.g;
    OutputBuffer[2 * channelSize + baseIndex] = normalized.b;
}
)";
}
#endif

// クラス状態チェック
YOLODetector::operator bool() const
{
    return m_->session != nullptr;
}

#ifdef _WIN32
// モデル読み込み（Windows 用：wchar_t*パス）
bool YOLODetector::loadModel(const wchar_t* model_path)
{
    try {
        // use_directml の設定に応じてセッションオプションを選択
        Ort::SessionOptions& opts = m_->use_directml ? m_->session_options_dml : m_->session_options;
        
        m_->session = std::make_unique<Ort::Session>(*m_->env, model_path, opts);
        m_->num_input_nodes = m_->session->GetInputCount();
        m_->num_output_nodes = m_->session->GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        m_->input_node_names.resize(m_->num_input_nodes);
        m_->input_node_name_strings.resize(m_->num_input_nodes);
        for (size_t i = 0; i < m_->num_input_nodes; i++) {
            auto input_name = m_->session->GetInputNameAllocated(i, allocator);
            m_->input_node_name_strings[i] = input_name.get();
            m_->input_node_names[i] = m_->input_node_name_strings[i].c_str();
        }

        m_->output_node_names.resize(m_->num_output_nodes);
        m_->output_node_name_strings.resize(m_->num_output_nodes);
        for (size_t i = 0; i < m_->num_output_nodes; i++) {
            auto output_name = m_->session->GetOutputNameAllocated(i, allocator);
            m_->output_node_name_strings[i] = output_name.get();
            m_->output_node_names[i] = m_->output_node_name_strings[i].c_str();
        }

        return true;
    }
    catch (const Ort::Exception&) {
        return false;
    }
}
#else
// モデル読み込み（Linux/macOS 用：char*パス）
bool YOLODetector::loadModel(const char* model_path)
{
    try {
        m_->session = std::make_unique<Ort::Session>(*m_->env, model_path, 
                                                      m_->session_options);
        m_->num_input_nodes = m_->session->GetInputCount();
        m_->num_output_nodes = m_->session->GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        m_->input_node_names.resize(m_->num_input_nodes);
        m_->input_node_name_strings.resize(m_->num_input_nodes);
        for (size_t i = 0; i < m_->num_input_nodes; i++) {
            auto input_name = m_->session->GetInputNameAllocated(i, allocator);
            m_->input_node_name_strings[i] = input_name.get();
            m_->input_node_names[i] = m_->input_node_name_strings[i].c_str();
        }

        m_->output_node_names.resize(m_->num_output_nodes);
        m_->output_node_name_strings.resize(m_->num_output_nodes);
        for (size_t i = 0; i < m_->num_output_nodes; i++) {
            auto output_name = m_->session->GetOutputNameAllocated(i, allocator);
            m_->output_node_name_strings[i] = output_name.get();
            m_->output_node_names[i] = m_->output_node_name_strings[i].c_str();
        }

        return true;
    }
    catch (const Ort::Exception&) {
        return false;
    }
}
#endif // _WIN32


std::optional<std::vector<YOLODetector::BoundingBox>> YOLODetector::inference(const cv::Mat& image,
                                      float conf_threshold)
{
    return inferenceImpl(image, 3, conf_threshold);
}

bool YOLODetector::initializeGpuPipeline(int width, int height)
{
#ifdef _WIN32
    constexpr int kModelWidth = 1280;
    constexpr int kModelHeight = 736;

    if (m_->gpu_pipeline_ready && m_->gpu_width == width && m_->gpu_height == height) {
        return true;
    }

    if (m_->gpu_pipeline_initialized && !m_->gpu_pipeline_ready) {
        return false;
    }

    releaseGpuResources();
    m_->gpu_pipeline_initialized = true;

    try {
        obs_log(LOG_INFO, "YOLO DML: initializing GPU preprocess pipeline for %dx%d", width, height);
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_->d3d12_device));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 device (hr=0x%08X)", hr);
            return false;
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        hr = m_->d3d12_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&m_->d3d12_queue));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 command queue (hr=0x%08X)", hr);
            return false;
        }

        hr = m_->d3d12_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                     IID_PPV_ARGS(&m_->d3d12_allocator));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 command allocator (hr=0x%08X)", hr);
            return false;
        }

        hr = m_->d3d12_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 m_->d3d12_allocator, nullptr,
                                                 IID_PPV_ARGS(&m_->d3d12_command_list));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 command list (hr=0x%08X)", hr);
            return false;
        }
        m_->d3d12_command_list->Close();

        hr = m_->d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_->d3d12_fence));
        if (FAILED(hr)) {
            return false;
        }
        m_->d3d12_fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!m_->d3d12_fence_event) {
            return false;
        }

        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                          D3D11_SDK_VERSION, &m_->d3d11_device, nullptr, &m_->d3d11_context);
        if (!m_->d3d11_device || !m_->d3d11_context) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D11 device/context");
            return false;
        }

        D3D11_TEXTURE2D_DESC tex_desc{};
        tex_desc.Width = width;
        tex_desc.Height = height;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        tex_desc.CPUAccessFlags = 0;
        tex_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        hr = m_->d3d11_device->CreateTexture2D(&tex_desc, nullptr, &m_->d3d11_shared_texture);
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create shared D3D11 texture (hr=0x%08X)", hr);
            return false;
        }

        hr = m_->d3d11_shared_texture->QueryInterface(__uuidof(IDXGIKeyedMutex),
                                                      reinterpret_cast<void**>(&m_->d3d11_keyed_mutex));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to query keyed mutex from shared texture (hr=0x%08X)", hr);
            return false;
        }

        IDXGIResource1* dxgi_resource = nullptr;
        hr = m_->d3d11_shared_texture->QueryInterface(__uuidof(IDXGIResource1),
                                                       reinterpret_cast<void**>(&dxgi_resource));
        if (FAILED(hr)) {
            return false;
        }
        hr = dxgi_resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                               nullptr, &m_->shared_handle);
        dxgi_resource->Release();
        if (FAILED(hr)) {
            return false;
        }

        D3D12_HEAP_PROPERTIES input_heap_props{};
        input_heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC input_tex_desc{};
        input_tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        input_tex_desc.Width = static_cast<UINT64>(width);
        input_tex_desc.Height = static_cast<UINT>(height);
        input_tex_desc.DepthOrArraySize = 1;
        input_tex_desc.MipLevels = 1;
        input_tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        input_tex_desc.SampleDesc.Count = 1;
        input_tex_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        input_tex_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = m_->d3d12_device->CreateCommittedResource(&input_heap_props, D3D12_HEAP_FLAG_NONE,
                                                      &input_tex_desc,
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                      nullptr, IID_PPV_ARGS(&m_->d3d12_shared_resource));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 input texture (hr=0x%08X)", hr);
            return false;
        }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 upload_buffer_size = 0;
        UINT num_rows = 0;
        UINT64 row_size_in_bytes = 0;
        m_->d3d12_device->GetCopyableFootprints(&input_tex_desc, 0, 1, 0, &footprint, &num_rows,
                                                &row_size_in_bytes, &upload_buffer_size);
        m_->d3d12_upload_footprint = footprint;

        D3D12_HEAP_PROPERTIES upload_heap_props{};
        upload_heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC upload_buffer_desc{};
        upload_buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        upload_buffer_desc.Width = upload_buffer_size;
        upload_buffer_desc.Height = 1;
        upload_buffer_desc.DepthOrArraySize = 1;
        upload_buffer_desc.MipLevels = 1;
        upload_buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
        upload_buffer_desc.SampleDesc.Count = 1;
        upload_buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = m_->d3d12_device->CreateCommittedResource(&upload_heap_props, D3D12_HEAP_FLAG_NONE,
                                                      &upload_buffer_desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ,
                                                      nullptr, IID_PPV_ARGS(&m_->d3d12_upload_buffer));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create D3D12 upload buffer (hr=0x%08X)", hr);
            return false;
        }

        hr = m_->d3d12_upload_buffer->Map(0, nullptr, &m_->d3d12_upload_mapped_ptr);
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to persistently map D3D12 upload buffer (hr=0x%08X)", hr);
            return false;
        }

        ID3DBlob* shader_blob = nullptr;
        ID3DBlob* error_blob = nullptr;
        hr = D3DCompile(kPreprocessShader, std::strlen(kPreprocessShader), nullptr, nullptr, nullptr,
                        "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shader_blob, &error_blob);
        if (FAILED(hr)) {
            if (error_blob) {
                obs_log(LOG_WARNING, "YOLO DML: shader compile failed: %s",
                        static_cast<const char*>(error_blob->GetBufferPointer()));
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

        ID3DBlob* serialized_root_sig = nullptr;
        ID3DBlob* root_sig_error_blob = nullptr;
        hr = D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &serialized_root_sig, &root_sig_error_blob);
        if (SUCCEEDED(hr)) {
            hr = m_->d3d12_device->CreateRootSignature(0, serialized_root_sig->GetBufferPointer(),
                                                       serialized_root_sig->GetBufferSize(),
                                                       IID_PPV_ARGS(&m_->d3d12_root_signature));
            serialized_root_sig->Release();
        }
        if (FAILED(hr)) {
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = m_->d3d12_root_signature;
        pso_desc.CS.pShaderBytecode = shader_blob->GetBufferPointer();
        pso_desc.CS.BytecodeLength = shader_blob->GetBufferSize();
        hr = m_->d3d12_device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&m_->d3d12_pipeline_state));
        shader_blob->Release();
        if (FAILED(hr)) {
            return false;
        }

        D3D12_HEAP_PROPERTIES cb_heap_props{};
        cb_heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC cb_desc{};
        cb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        cb_desc.Width = (sizeof(GPUPreprocessConfig) + 255) & ~255;
        cb_desc.Height = 1;
        cb_desc.DepthOrArraySize = 1;
        cb_desc.MipLevels = 1;
        cb_desc.Format = DXGI_FORMAT_UNKNOWN;
        cb_desc.SampleDesc.Count = 1;
        cb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = m_->d3d12_device->CreateCommittedResource(&cb_heap_props, D3D12_HEAP_FLAG_NONE, &cb_desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&m_->d3d12_constant_buffer));
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to create descriptor heap (hr=0x%08X)", hr);
            return false;
        }

        hr = m_->d3d12_constant_buffer->Map(0, nullptr, &m_->d3d12_constant_mapped_ptr);
        if (FAILED(hr)) {
            obs_log(LOG_WARNING, "YOLO DML: failed to persistently map constant buffer (hr=0x%08X)", hr);
            return false;
        }

        const uint32_t dst_w = kModelWidth;
        const uint32_t dst_h = kModelHeight;
        const uint32_t uav_size = dst_w * dst_h * 3 * sizeof(float);
        D3D12_HEAP_PROPERTIES uav_heap_props{};
        uav_heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC uav_desc{};
        uav_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uav_desc.Width = uav_size;
        uav_desc.Height = 1;
        uav_desc.DepthOrArraySize = 1;
        uav_desc.MipLevels = 1;
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.SampleDesc.Count = 1;
        uav_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        uav_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr = m_->d3d12_device->CreateCommittedResource(&uav_heap_props, D3D12_HEAP_FLAG_NONE, &uav_desc,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&m_->d3d12_uav_buffer));
        if (FAILED(hr)) {
            return false;
        }

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.NumDescriptors = 2;
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = m_->d3d12_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&m_->d3d12_descriptor_heap));
        if (FAILED(hr)) {
            return false;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE heap_start = m_->d3d12_descriptor_heap->GetCPUDescriptorHandleForHeapStart();
        const UINT increment = m_->d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE srv_handle = heap_start;
        D3D12_CPU_DESCRIPTOR_HANDLE uav_handle = heap_start;
        uav_handle.ptr += increment;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Texture2D.MipLevels = 1;
        srv_desc.Texture2D.MostDetailedMip = 0;
        m_->d3d12_device->CreateShaderResourceView(m_->d3d12_shared_resource, &srv_desc, srv_handle);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_view_desc{};
        uav_view_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_view_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_view_desc.Buffer.FirstElement = 0;
        uav_view_desc.Buffer.NumElements = dst_w * dst_h * 3;
        uav_view_desc.Buffer.StructureByteStride = sizeof(float);
        m_->d3d12_device->CreateUnorderedAccessView(m_->d3d12_uav_buffer, nullptr, &uav_view_desc, uav_handle);

        m_->gpu_width = width;
        m_->gpu_height = height;

        const auto& ort_api = Ort::GetApi();
        const OrtDmlApi* dml_api = nullptr;
        OrtStatus* status = ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api));
        if (status != nullptr) {
            ort_api.ReleaseStatus(status);
            return false;
        }

        status = dml_api->CreateGPUAllocationFromD3DResource(m_->d3d12_uav_buffer, &m_->dml_gpu_allocation);
        if (status != nullptr) {
            ort_api.ReleaseStatus(status);
            obs_log(LOG_WARNING, "YOLO DML: failed to create ORT GPU allocation from D3D resource");
            return false;
        }

        m_->gpu_pipeline_ready = true;
        obs_log(LOG_INFO, "YOLO DML: GPU preprocess pipeline ready for %dx%d", width, height);
        return true;
    } catch (const std::exception&) {
        return false;
    }
#else
    return false;
#endif
}

#ifdef _WIN32
std::optional<Ort::Value> YOLODetector::runGpuPreprocessFromSourceResource(void *source_resource_ptr, int width, int height)
{
    auto *source_resource = reinterpret_cast<ID3D12Resource *>(source_resource_ptr);
    if (!m_ || !source_resource) {
        return std::nullopt;
    }

    if (m_->d3d12_command_list == nullptr || m_->d3d12_descriptor_heap == nullptr ||
        m_->d3d12_root_signature == nullptr || m_->d3d12_pipeline_state == nullptr ||
        m_->d3d12_constant_buffer == nullptr || m_->d3d12_uav_buffer == nullptr) {
        return std::nullopt;
    }

    GPUPreprocessConfig cb_data{};
    constexpr int kModelWidth = 1280;
    constexpr int kModelHeight = 736;
    float padX = 0.0f;
    float padY = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;

    if (static_cast<float>(width) / static_cast<float>(height) >= static_cast<float>(kModelWidth) / static_cast<float>(kModelHeight)) {
        const int resizedWidth = kModelWidth;
        const int resizedHeight = static_cast<int>(height * kModelWidth / width);
        padY = (kModelHeight - resizedHeight) / 2.0f;
        scaleX = static_cast<float>(resizedWidth) / static_cast<float>(width);
        scaleY = static_cast<float>(resizedHeight) / static_cast<float>(height);
        cb_data.unpad_width = static_cast<uint32_t>(resizedWidth);
        cb_data.unpad_height = static_cast<uint32_t>(resizedHeight);
    } else {
        const int resizedHeight = kModelHeight;
        const int resizedWidth = static_cast<int>(width * kModelHeight / height);
        padX = (kModelWidth - resizedWidth) / 2.0f;
        scaleX = static_cast<float>(resizedWidth) / static_cast<float>(width);
        scaleY = static_cast<float>(resizedHeight) / static_cast<float>(height);
        cb_data.unpad_width = static_cast<uint32_t>(resizedWidth);
        cb_data.unpad_height = static_cast<uint32_t>(resizedHeight);
    }

    cb_data.src_width = static_cast<uint32_t>(width);
    cb_data.src_height = static_cast<uint32_t>(height);
    cb_data.dst_width = static_cast<uint32_t>(kModelWidth);
    cb_data.dst_height = static_cast<uint32_t>(kModelHeight);
    cb_data.pad_x = static_cast<uint32_t>(padX);
    cb_data.pad_y = static_cast<uint32_t>(padY);
    cb_data.mean_r = 0.0f;
    cb_data.mean_g = 0.0f;
    cb_data.mean_b = 0.0f;
    cb_data.std_r = 1.0f;
    cb_data.std_g = 1.0f;
    cb_data.std_b = 1.0f;

    void *mapped = nullptr;
    if (m_->d3d12_constant_buffer->Map(0, nullptr, &mapped) != S_OK || !mapped) {
        obs_log(LOG_WARNING, "YOLO DML: failed to map constant buffer for preprocessing");
        return std::nullopt;
    }
    std::memcpy(mapped, &cb_data, sizeof(cb_data));
    m_->d3d12_constant_buffer->Unmap(0, nullptr);

    HRESULT hr = m_->d3d12_allocator->Reset();
    if (FAILED(hr)) {
        return std::nullopt;
    }
    hr = m_->d3d12_command_list->Reset(m_->d3d12_allocator, m_->d3d12_pipeline_state);
    if (FAILED(hr)) {
        return std::nullopt;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE heap_start_cpu = m_->d3d12_descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE heap_start_gpu = m_->d3d12_descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srv_handle = heap_start_cpu;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_handle = heap_start_gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu_handle = heap_start_gpu;
    uav_gpu_handle.ptr += m_->d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    m_->d3d12_device->CreateShaderResourceView(source_resource, &srv_desc, srv_handle);

    m_->d3d12_command_list->SetComputeRootSignature(m_->d3d12_root_signature);
    m_->d3d12_command_list->SetPipelineState(m_->d3d12_pipeline_state);
    ID3D12DescriptorHeap *heaps[] = {m_->d3d12_descriptor_heap};
    m_->d3d12_command_list->SetDescriptorHeaps(1, heaps);
    m_->d3d12_command_list->SetComputeRootConstantBufferView(0, m_->d3d12_constant_buffer->GetGPUVirtualAddress());
    m_->d3d12_command_list->SetComputeRootDescriptorTable(1, srv_gpu_handle);
    m_->d3d12_command_list->SetComputeRootDescriptorTable(2, uav_gpu_handle);
    const UINT dispatch_x = static_cast<UINT>((kModelWidth + 15) / 16);
    const UINT dispatch_y = static_cast<UINT>((kModelHeight + 15) / 16);
    m_->d3d12_command_list->Dispatch(dispatch_x, dispatch_y, 1);
    m_->d3d12_command_list->Close();

    ID3D12CommandList *lists[] = {m_->d3d12_command_list};
    m_->d3d12_queue->ExecuteCommandLists(1, lists);
    m_->d3d12_fence_value++;
    m_->d3d12_queue->Signal(m_->d3d12_fence, m_->d3d12_fence_value);
    if (m_->d3d12_fence->GetCompletedValue() < m_->d3d12_fence_value) {
        m_->d3d12_fence->SetEventOnCompletion(m_->d3d12_fence_value, m_->d3d12_fence_event);
        WaitForSingleObject(m_->d3d12_fence_event, INFINITE);
    }

    if (m_->dml_gpu_allocation == nullptr || !m_->dml_memory_info) {
        obs_log(LOG_WARNING, "YOLO DML: missing ORT GPU allocation for bound input");
        return std::nullopt;
    }

    std::vector<int64_t> input_tensor_shape = {1, 3, static_cast<int64_t>(kModelHeight), static_cast<int64_t>(kModelWidth)};
    const size_t tensor_byte_size = static_cast<size_t>(kModelWidth) * static_cast<size_t>(kModelHeight) * 3 * sizeof(float);

    return Ort::Value::CreateTensor(
        *m_->dml_memory_info,
        m_->dml_gpu_allocation,
        tensor_byte_size,
        input_tensor_shape.data(),
        input_tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}
#endif

std::optional<Ort::Value> YOLODetector::runGpuPreprocessToOrtValue(const cv::Mat& image)
{
#ifdef _WIN32
    if (!m_->use_directml || !m_->gpu_zero_copy_enabled || image.empty()) {
        return std::nullopt;
    }

    const int width = image.cols;
    const int height = image.rows;
    if (width <= 0 || height <= 0) {
        return std::nullopt;
    }

    if (!initializeGpuPipeline(width, height)) {
        obs_log(LOG_WARNING, "YOLO DML: GPU preprocessing pipeline initialization failed; falling back to CPU preprocessing");
        return std::nullopt;
    }

    if (!m_->d3d12_shared_resource || !m_->d3d12_upload_buffer) {
        return std::nullopt;
    }

    cv::Mat upload_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, upload_image, cv::COLOR_BGR2BGRA);
    } else {
        upload_image = image;
    }

    if (!m_->d3d12_upload_mapped_ptr) {
        obs_log(LOG_WARNING, "YOLO DML: upload buffer is not mapped");
        return std::nullopt;
    }
    const UINT row_pitch = m_->d3d12_upload_footprint.Footprint.RowPitch;
    const UINT src_row_bytes = static_cast<UINT>(width) * 4u;
    uint8_t* dst = static_cast<uint8_t*>(m_->d3d12_upload_mapped_ptr);
    const uint8_t* src = upload_image.data;
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * row_pitch,
                    src + static_cast<size_t>(y) * upload_image.step[0],
                    src_row_bytes);
    }

    if (m_->d3d12_command_list == nullptr || m_->d3d12_descriptor_heap == nullptr ||
        m_->d3d12_root_signature == nullptr || m_->d3d12_pipeline_state == nullptr ||
        m_->d3d12_constant_buffer == nullptr || m_->d3d12_uav_buffer == nullptr) {
        return std::nullopt;
    }

    GPUPreprocessConfig cb_data{};
    constexpr int kModelWidth = 1280;
    constexpr int kModelHeight = 736;
    float padX = 0.0f;
    float padY = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;

    if (static_cast<float>(width) / static_cast<float>(height) >= static_cast<float>(kModelWidth) / static_cast<float>(kModelHeight)) {
        const int resizedWidth = kModelWidth;
        const int resizedHeight = static_cast<int>(height * kModelWidth / width);
        padY = (kModelHeight - resizedHeight) / 2.0f;
        scaleX = static_cast<float>(resizedWidth) / static_cast<float>(width);
        scaleY = static_cast<float>(resizedHeight) / static_cast<float>(height);
        cb_data.unpad_width = static_cast<uint32_t>(resizedWidth);
        cb_data.unpad_height = static_cast<uint32_t>(resizedHeight);
    } else {
        const int resizedHeight = kModelHeight;
        const int resizedWidth = static_cast<int>(width * kModelHeight / height);
        padX = (kModelWidth - resizedWidth) / 2.0f;
        scaleX = static_cast<float>(resizedWidth) / static_cast<float>(width);
        scaleY = static_cast<float>(resizedHeight) / static_cast<float>(height);
        cb_data.unpad_width = static_cast<uint32_t>(resizedWidth);
        cb_data.unpad_height = static_cast<uint32_t>(resizedHeight);
    }

    cb_data.src_width = static_cast<uint32_t>(width);
    cb_data.src_height = static_cast<uint32_t>(height);
    cb_data.dst_width = static_cast<uint32_t>(kModelWidth);
    cb_data.dst_height = static_cast<uint32_t>(kModelHeight);
    cb_data.pad_x = static_cast<uint32_t>(padX);
    cb_data.pad_y = static_cast<uint32_t>(padY);
    cb_data.mean_r = 0.0f;
    cb_data.mean_g = 0.0f;
    cb_data.mean_b = 0.0f;
    cb_data.std_r = 1.0f;
    cb_data.std_g = 1.0f;
    cb_data.std_b = 1.0f;

    if (!m_->d3d12_constant_mapped_ptr) {
        obs_log(LOG_WARNING, "YOLO DML: constant buffer is not mapped");
        return std::nullopt;
    }
    std::memcpy(m_->d3d12_constant_mapped_ptr, &cb_data, sizeof(cb_data));

    HRESULT hr = m_->d3d12_allocator->Reset();
    if (FAILED(hr)) {
        return std::nullopt;
    }
    hr = m_->d3d12_command_list->Reset(m_->d3d12_allocator, m_->d3d12_pipeline_state);
    if (FAILED(hr)) {
        return std::nullopt;
    }

    m_->d3d12_command_list->SetComputeRootSignature(m_->d3d12_root_signature);
    m_->d3d12_command_list->SetPipelineState(m_->d3d12_pipeline_state);
    ID3D12DescriptorHeap* heaps[] = {m_->d3d12_descriptor_heap};
    m_->d3d12_command_list->SetDescriptorHeaps(1, heaps);
    D3D12_GPU_DESCRIPTOR_HANDLE descriptor_heap_start = m_->d3d12_descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_handle = descriptor_heap_start;
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu_handle = descriptor_heap_start;
    uav_gpu_handle.ptr += m_->d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_RESOURCE_BARRIER pre_copy_barrier{};
    pre_copy_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    pre_copy_barrier.Transition.pResource = m_->d3d12_shared_resource;
    pre_copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    pre_copy_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    pre_copy_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_->d3d12_command_list->ResourceBarrier(1, &pre_copy_barrier);

    D3D12_TEXTURE_COPY_LOCATION dst_location{};
    dst_location.pResource = m_->d3d12_shared_resource;
    dst_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_location.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src_location{};
    src_location.pResource = m_->d3d12_upload_buffer;
    src_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_location.PlacedFootprint = m_->d3d12_upload_footprint;

    m_->d3d12_command_list->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, nullptr);

    D3D12_RESOURCE_BARRIER post_copy_barrier{};
    post_copy_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    post_copy_barrier.Transition.pResource = m_->d3d12_shared_resource;
    post_copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    post_copy_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    post_copy_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_->d3d12_command_list->ResourceBarrier(1, &post_copy_barrier);

    m_->d3d12_command_list->SetComputeRootConstantBufferView(0, m_->d3d12_constant_buffer->GetGPUVirtualAddress());
    m_->d3d12_command_list->SetComputeRootDescriptorTable(1, srv_gpu_handle);
    m_->d3d12_command_list->SetComputeRootDescriptorTable(2, uav_gpu_handle);
    const UINT dispatch_x = static_cast<UINT>((kModelWidth + 15) / 16);
    const UINT dispatch_y = static_cast<UINT>((kModelHeight + 15) / 16);
    m_->d3d12_command_list->Dispatch(dispatch_x, dispatch_y, 1);
    m_->d3d12_command_list->Close();

    m_->d3d12_queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&m_->d3d12_command_list));
    obs_log(LOG_DEBUG, "YOLO DML: dispatched compute shader for %dx%d", width, height);
    m_->d3d12_fence_value++;
    m_->d3d12_queue->Signal(m_->d3d12_fence, m_->d3d12_fence_value);
    if (m_->d3d12_fence->GetCompletedValue() < m_->d3d12_fence_value) {
        m_->d3d12_fence->SetEventOnCompletion(m_->d3d12_fence_value, m_->d3d12_fence_event);
        WaitForSingleObject(m_->d3d12_fence_event, INFINITE);
    }

    if (m_->dml_gpu_allocation == nullptr || !m_->dml_memory_info) {
        obs_log(LOG_WARNING, "YOLO DML: missing ORT GPU allocation for bound input");
        return std::nullopt;
    }

    std::vector<int64_t> input_tensor_shape = {
        1,
        3,
        static_cast<int64_t>(kModelHeight),
        static_cast<int64_t>(kModelWidth),
    };
    const size_t tensor_byte_size = static_cast<size_t>(kModelWidth) * static_cast<size_t>(kModelHeight) * 3 * sizeof(float);

    return Ort::Value::CreateTensor(
        *m_->dml_memory_info,
        m_->dml_gpu_allocation,
        tensor_byte_size,
        input_tensor_shape.data(),
        input_tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
#else
    return std::nullopt;
#endif
}

void YOLODetector::releaseGpuResources()
{
#ifdef _WIN32
    if (m_->dml_gpu_allocation) {
        const auto& ort_api = Ort::GetApi();
        const OrtDmlApi* dml_api = nullptr;
        OrtStatus* status = ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                            reinterpret_cast<const void**>(&dml_api));
        if (status == nullptr && dml_api != nullptr) {
            dml_api->FreeGPUAllocation(m_->dml_gpu_allocation);
        }
        m_->dml_gpu_allocation = nullptr;
    }
    if (m_->d3d12_fence_event) {
        CloseHandle(m_->d3d12_fence_event);
        m_->d3d12_fence_event = nullptr;
    }
    if (m_->d3d11_context) {
        m_->d3d11_context->Release();
        m_->d3d11_context = nullptr;
    }
    if (m_->d3d11_device) {
        m_->d3d11_device->Release();
        m_->d3d11_device = nullptr;
    }
    if (m_->d3d11_keyed_mutex) {
        m_->d3d11_keyed_mutex->Release();
        m_->d3d11_keyed_mutex = nullptr;
    }
    if (m_->d3d11_shared_texture) {
        m_->d3d11_shared_texture->Release();
        m_->d3d11_shared_texture = nullptr;
    }
    if (m_->d3d12_upload_buffer && m_->d3d12_upload_mapped_ptr) {
        m_->d3d12_upload_buffer->Unmap(0, nullptr);
        m_->d3d12_upload_mapped_ptr = nullptr;
    }
    if (m_->d3d12_constant_buffer && m_->d3d12_constant_mapped_ptr) {
        m_->d3d12_constant_buffer->Unmap(0, nullptr);
        m_->d3d12_constant_mapped_ptr = nullptr;
    }
    if (m_->d3d12_command_list) {
        m_->d3d12_command_list->Release();
        m_->d3d12_command_list = nullptr;
    }
    if (m_->d3d12_allocator) {
        m_->d3d12_allocator->Release();
        m_->d3d12_allocator = nullptr;
    }
    if (m_->d3d12_queue) {
        m_->d3d12_queue->Release();
        m_->d3d12_queue = nullptr;
    }
    if (m_->d3d12_shared_resource) {
        m_->d3d12_shared_resource->Release();
        m_->d3d12_shared_resource = nullptr;
    }
    if (m_->d3d12_upload_buffer) {
        m_->d3d12_upload_buffer->Release();
        m_->d3d12_upload_buffer = nullptr;
    }
    if (m_->d3d12_uav_buffer) {
        m_->d3d12_uav_buffer->Release();
        m_->d3d12_uav_buffer = nullptr;
    }
    if (m_->d3d12_constant_buffer) {
        m_->d3d12_constant_buffer->Release();
        m_->d3d12_constant_buffer = nullptr;
    }
    if (m_->d3d12_descriptor_heap) {
        m_->d3d12_descriptor_heap->Release();
        m_->d3d12_descriptor_heap = nullptr;
    }
    if (m_->d3d12_root_signature) {
        m_->d3d12_root_signature->Release();
        m_->d3d12_root_signature = nullptr;
    }
    if (m_->d3d12_pipeline_state) {
        m_->d3d12_pipeline_state->Release();
        m_->d3d12_pipeline_state = nullptr;
    }
    if (m_->d3d12_fence) {
        m_->d3d12_fence->Release();
        m_->d3d12_fence = nullptr;
    }
    if (m_->d3d12_device) {
        m_->d3d12_device->Release();
        m_->d3d12_device = nullptr;
    }
    m_->gpu_pipeline_ready = false;
    m_->gpu_pipeline_initialized = false;
    m_->gpu_width = 0;
    m_->gpu_height = 0;
#endif
}

std::optional<std::vector<YOLODetector::BoundingBox>> YOLODetector::inferenceBGRA(const cv::Mat& image,
                                       float conf_threshold)
{
    return inferenceImpl(image, 4, conf_threshold);
}

#ifdef _WIN32
std::optional<std::vector<YOLODetector::BoundingBox>> YOLODetector::inferenceFromSharedHandle(
    HANDLE sharedHandle,
    uint32_t width,
    uint32_t height,
    float conf_threshold)
{
    if (!m_->use_directml || !m_->gpu_zero_copy_enabled || !sharedHandle || width == 0 || height == 0) {
        return std::nullopt;
    }

    if (!m_->session) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    ID3D12Resource *source_resource = nullptr;
    HRESULT hr = m_->d3d12_device->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&source_resource));
    if (FAILED(hr) || !source_resource) {
        obs_log(LOG_WARNING, "YOLO DML: failed to open shared render texture handle (hr=0x%08X)", hr);
        return std::nullopt;
    }

    auto gpu_input_tensor = runGpuPreprocessFromSourceResource(source_resource, static_cast<int>(width), static_cast<int>(height));
    source_resource->Release();
    if (!gpu_input_tensor.has_value()) {
        return std::nullopt;
    }

    std::vector<Ort::Value> output_tensors;
    try {
        if (m_->input_node_names.empty() || m_->output_node_names.empty()) {
            return std::vector<BoundingBox>();
        }

        output_tensors = m_->session->Run(
            Ort::RunOptions{nullptr},
            m_->input_node_names.data(),
            &*gpu_input_tensor, 1,
            m_->output_node_names.data(),
            m_->num_output_nodes);
    } catch (const Ort::Exception &e) {
        obs_log(LOG_WARNING, "YOLO inference run failed: %s", e.what());
        return std::vector<BoundingBox>();
    }

    std::vector<BoundingBox> bboxes;
    if (output_tensors.empty()) {
        return bboxes;
    }

    auto info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    if (shape.size() != 3) {
        return bboxes;
    }

    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];
    int64_t detections = shape[1];
    int64_t per_det = shape[2];
    if (shape[2] == 6 || shape[2] == 7) {
        detections = shape[1];
        per_det = shape[2];
    } else if (shape[1] == 6 || shape[1] == 7) {
        detections = shape[2];
        per_det = shape[1];
    }

    if (shape[2] == 6 || shape[2] == 7) {
        detections = shape[1];
        per_det = shape[2];
    } else if (shape[1] == 6 || shape[1] == 7) {
        detections = shape[2];
        per_det = shape[1];
    }

    const float *data = output_tensors.front().GetTensorData<float>();
    const float scaleX = static_cast<float>(width) / 1280.0f;
    const float scaleY = static_cast<float>(height) / 736.0f;
    const float padX = 0.0f;
    const float padY = 0.0f;
    const float score_threshold = conf_threshold;
    bboxes = [&]() {
        std::vector<BoundingBox> boxes;
        if (shape.size() != 3) {
            return boxes;
        }
        for (int64_t i = 0; i < detections; ++i) {
            const float *det = data + i * per_det;
            if (per_det < 6) {
                continue;
            }
            float score = det[4];
            if (score <= score_threshold) {
                continue;
            }
            float x1_model = det[0];
            float y1_model = det[1];
            float x2_model = det[2];
            float y2_model = det[3];
            int class_id = static_cast<int>(det[5]);
            float x1 = (x1_model - padX) / scaleX;
            float y1 = (y1_model - padY) / scaleY;
            float x2 = (x2_model - padX) / scaleX;
            float y2 = (y2_model - padY) / scaleY;
            float leftVal = (x1 < x2) ? x1 : x2;
            float topVal = (y1 < y2) ? y1 : y2;
            int left = static_cast<int>(std::round(std::max<float>(0.0f, leftVal)));
            int top = static_cast<int>(std::round(std::max<float>(0.0f, topVal)));
            int width_box = static_cast<int>(std::round(std::abs(x2 - x1)));
            int height_box = static_cast<int>(std::round(std::abs(y2 - y1)));
            if (width_box <= 0 || height_box <= 0) {
                continue;
            }
            BoundingBox bbox;
            bbox.score = score;
            bbox.index = class_id;
            bbox.rect = cv::Rect(left, top, width_box, height_box);
            boxes.push_back(bbox);
        }
        return boxes;
    }();

    return bboxes;
}
#endif

// 推論実行
std::optional<std::vector<YOLODetector::BoundingBox>> YOLODetector::inferenceImpl(
    const cv::Mat& image, int expected_channels, float conf_threshold)
{
    if (image.empty() || image.rows == 0 || image.cols == 0) {
        return std::vector<BoundingBox>();
    }

    if ((expected_channels == 3 && image.type() != CV_8UC3) ||
        (expected_channels == 4 && image.type() != CV_8UC4)) {
        return std::nullopt;
    }

    if (!m_->session) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const int N = 1;
    const int C = 3;
    const int W = 1280;
    const int H = 736;

    float targetRatio = static_cast<float>(W) / H;
    float imgRatio = static_cast<float>(image.cols) / image.rows;

    cv::Mat &resizedImg = m_->resized_img;
    float padX = 0;
    float padY = 0;
    float scaleX = 1.0f;
    float scaleY = 1.0f;

    auto run_and_decode = [&](const Ort::Value &input_tensor) -> std::optional<std::vector<BoundingBox>> {
        std::vector<Ort::Value> output_tensors;
        try {
            if (m_->input_node_names.empty() || m_->output_node_names.empty()) {
                return std::vector<BoundingBox>();
            }

            output_tensors = m_->session->Run(
                Ort::RunOptions{nullptr},
                m_->input_node_names.data(),
                &input_tensor, 1,
                m_->output_node_names.data(),
                m_->num_output_nodes);
        } catch (const Ort::Exception &e) {
            obs_log(LOG_WARNING, "YOLO inference run failed: %s", e.what());
            return std::vector<BoundingBox>();
        }

        std::vector<BoundingBox> bboxes;
        if (output_tensors.empty()) {
            return bboxes;
        }

        auto info = output_tensors.front().GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        if (shape.size() != 3) {
            return bboxes;
        }

        int64_t dim1 = shape[1];
        int64_t dim2 = shape[2];
        int64_t detections = 0;
        int64_t per_det = 0;

        if (dim2 == 6) {
            detections = dim1;
            per_det = dim2;
        } else if (dim1 == 6) {
            detections = dim2;
            per_det = dim1;
        } else {
            per_det = static_cast<int>(shape[1]);
            detections = static_cast<int>(shape[2]);
        }

        float const *output_tensor = output_tensors[0].GetTensorData<float>();
        const float *batch_base = output_tensor;
        const float score_threshold = conf_threshold;

        for (int64_t i = 0; i < detections; ++i) {
            const float *det = batch_base + i * per_det;
            if (per_det < 6) {
                continue;
            }

            float score = det[4];
            if (score <= score_threshold) {
                continue;
            }

            float x1_model = det[0];
            float y1_model = det[1];
            float x2_model = det[2];
            float y2_model = det[3];
            int class_id = static_cast<int>(det[5]);

            float x1 = (x1_model - padX) / scaleX;
            float y1 = (y1_model - padY) / scaleY;
            float x2 = (x2_model - padX) / scaleX;
            float y2 = (y2_model - padY) / scaleY;

            float leftVal = (x1 < x2) ? x1 : x2;
            float topVal = (y1 < y2) ? y1 : y2;
            int left = static_cast<int>(std::round(std::max<float>(0.0f, leftVal)));
            int top = static_cast<int>(std::round(std::max<float>(0.0f, topVal)));
            int width = static_cast<int>(std::round(std::abs(x2 - x1)));
            int height = static_cast<int>(std::round(std::abs(y2 - y1)));

            if (width <= 0 || height <= 0) {
                continue;
            }

            BoundingBox bbox;
            bbox.score = score;
            bbox.index = class_id;
            bbox.rect = cv::Rect(left, top, width, height);
            bboxes.push_back(bbox);
        }

        return bboxes;
    };
    if (imgRatio >= targetRatio) {
        int newWidth = W;
        int newHeight = static_cast<int>(image.rows * W / image.cols);
        cv::resize(image, resizedImg, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        padY = (H - newHeight) / 2.0f;
        scaleX = static_cast<float>(W) / image.cols;
        scaleY = static_cast<float>(newHeight) / image.rows;
    } else {
        int newHeight = H;
        int newWidth = static_cast<int>(image.cols * H / image.rows);
        cv::resize(image, resizedImg, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        padX = (W - newWidth) / 2.0f;
        scaleX = static_cast<float>(newWidth) / image.cols;
        scaleY = static_cast<float>(H) / image.rows;
    }

    const int paddedType = expected_channels == 4 ? CV_8UC4 : CV_8UC3;
    if (m_->padded_img.empty() || m_->padded_img.cols != W || m_->padded_img.rows != H ||
        m_->padded_img.type() != paddedType) {
        m_->padded_img.create(H, W, paddedType);
    }
    cv::Mat &paddedImg = m_->padded_img;
    if (expected_channels == 4) {
        paddedImg.setTo(cv::Scalar(0, 0, 0, 255));
    } else {
        paddedImg.setTo(cv::Scalar(0, 0, 0));
    }
    resizedImg.copyTo(paddedImg(cv::Rect(static_cast<int>(padX), static_cast<int>(padY),
                                          resizedImg.cols, resizedImg.rows)));

    resizeScales = std::max(scaleX, scaleY);

    if (expected_channels == 4 && m_->use_directml && m_->gpu_zero_copy_enabled) {
        auto gpu_input_tensor = runGpuPreprocessToOrtValue(image);
        if (gpu_input_tensor.has_value()) {
            return run_and_decode(*gpu_input_tensor);
        }
        obs_log(LOG_WARNING, "YOLO DML: GPU zero-copy input failed; falling back to CPU preprocessing");
    }

    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t inputSize = static_cast<size_t>(C) * plane;
    if (m_->input_tensor_buffer.size() != inputSize) {
        m_->input_tensor_buffer.resize(inputSize);
    }
    float *inputData = m_->input_tensor_buffer.data();
    const float norm = 1.0f / 255.0f;
    if (expected_channels == 4) {
        for (int y = 0; y < H; ++y) {
            const cv::Vec4b *row = paddedImg.ptr<cv::Vec4b>(y);
            for (int x = 0; x < W; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                const cv::Vec4b &bgra = row[x];
                inputData[idx] = static_cast<float>(bgra[2]) * norm;
                inputData[idx + plane] = static_cast<float>(bgra[1]) * norm;
                inputData[idx + (2 * plane)] = static_cast<float>(bgra[0]) * norm;
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            const cv::Vec3b *row = paddedImg.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
                const cv::Vec3b &bgr = row[x];
                inputData[idx] = static_cast<float>(bgr[2]) * norm;
                inputData[idx + plane] = static_cast<float>(bgr[1]) * norm;
                inputData[idx + (2 * plane)] = static_cast<float>(bgr[0]) * norm;
            }
        }
    }

    std::vector<int64_t> input_tensor_shape = {static_cast<int64_t>(N), static_cast<int64_t>(C),
                                               static_cast<int64_t>(H), static_cast<int64_t>(W)};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *m_->memory_info,
        inputData,
        inputSize,
        input_tensor_shape.data(),
        input_tensor_shape.size());

    return run_and_decode(input_tensor);
}

// Object 構造体への変換
std::vector<Object> YOLODetector::convertToObjects(const std::vector<BoundingBox>& bboxes)
{
    std::vector<Object> objects;
    objects.reserve(bboxes.size());

    for (const auto& bbox : bboxes) {
        Object obj;
        obj.rect = cv::Rect_<float>(static_cast<float>(bbox.rect.x), 
                                    static_cast<float>(bbox.rect.y),
                                    static_cast<float>(bbox.rect.width),
                                    static_cast<float>(bbox.rect.height));
        obj.label = bbox.index;          // 単純増加インデックスを label に代入
        obj.prob = bbox.score;           // 信頼度スコアを prob に代入
        obj.id = 0;                      // ID は後から設定（トラッキング用）
        // obj.unseenFrames = 0;            // 未検出フレームカウント
        // kf(KalmanFilter) は必要に応じて初期化
        
        objects.push_back(obj);
    }

    return objects;
}

// デバイス設定
void YOLODetector::setUseGPU(bool useGPU)
{
    m_->use_directml = useGPU;
}

void YOLODetector::setGpuZeroCopyEnabled(bool enabled)
{
    m_->gpu_zero_copy_enabled = enabled;
}

void YOLODetector::setNumThreads(uint32_t numThreads)
{
    m_->num_threads = std::max<uint32_t>(1, numThreads);
    m_->session_options.SetIntraOpNumThreads(static_cast<int>(m_->num_threads));
#ifdef _WIN32
    m_->session_options_dml.SetIntraOpNumThreads(static_cast<int>(m_->num_threads));
#endif
}

// DirectML 初期化（Windows のみ）
bool YOLODetector::initializeDirectML()
{
#ifdef _WIN32
    try {
        m_->session_options_dml = Ort::SessionOptions();
        
        m_->session_options_dml.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_->session_options_dml.SetLogSeverityLevel(3);
        m_->session_options_dml.SetIntraOpNumThreads(static_cast<int>(m_->num_threads));

        // ONNX Runtime の API を使用して DML エクスプローラーを取得・追加
        const auto& api = Ort::GetApi();
        OrtDmlApi* dmlApi = nullptr;
        
        Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, 
                                                       (const void**)&dmlApi));
        
        if (dmlApi != nullptr) {
            // DML エクスプローラーを追加
            Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
                m_->session_options_dml, 0));
            
            obs_log(LOG_INFO, "DirectML execution provider initialized successfully");
        } else {
            obs_log(LOG_WARNING, "DML API not available");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        obs_log(LOG_WARNING, "DirectML not available: %s", e.what());
        return false;
    }
#else
    // Windows のみの機能
    return false;
#endif
}
