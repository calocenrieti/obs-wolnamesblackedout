#include "yolodetector/YOLODetector.h"
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <plugin-support.h>

#ifdef _WIN32
#include <windows.h>
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
    
    // DirectML 用
#ifdef _WIN32
    Ort::SessionOptions session_options_dml;
#endif
};

// コンストラクタ
YOLODetector::YOLODetector()
    : m_(std::make_unique<Private>())
    , resizeScales(1.0f)
{
    try {
        m_->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
                                       OrtMemType::OrtMemTypeDefault));

        // ONNX Runtime 初期化（CPU のみサポート）
        m_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");
        
        m_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_->session_options.SetLogSeverityLevel(3);
        m_->session_options.SetIntraOpNumThreads(1);
    }
    catch (const std::exception& e) {
        // 初期化エラーは内部で吸収
    }
}

// デストラクタ
YOLODetector::~YOLODetector() = default;

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

// 前処理（リサイズ、正規化など）
bool YOLODetector::PreProcess2(const cv::Mat& iImg, int targetWidth, int targetHeight, 
                               cv::Mat& oImg)
{
    // 目標のアスペクト比
    float targetRatio = static_cast<float>(targetWidth) / targetHeight;
    // 入力画像のアスペクト比
    float imgRatio = static_cast<float>(iImg.cols) / iImg.rows;

    // 横長、またはアスペクト比が同じ場合：横幅基準でリサイズ
    if (imgRatio >= targetRatio) {
        resizeScales = iImg.cols / static_cast<float>(targetWidth);
        int newHeight = static_cast<int>(iImg.rows / resizeScales);
        cv::resize(iImg, oImg, cv::Size(targetWidth, std::max<int>(1, newHeight)), 0, 0, 
                   cv::INTER_LINEAR);
    }
    // 縦長の場合：縦幅基準でリサイズ
    else {
        resizeScales = iImg.rows / static_cast<float>(targetHeight);
        int newWidth = static_cast<int>(iImg.cols / resizeScales);
        cv::resize(iImg, oImg, cv::Size(std::max<int>(1, newWidth), targetHeight), 0, 0, 
                   cv::INTER_LINEAR);
    }

    // 出力テンソル（背景は 0 で埋める＝黒）の作成
    cv::Mat tempImg = cv::Mat::zeros(targetHeight, targetWidth, CV_8UC3);

    // リサイズ済み画像を左上にコピー
    if (!oImg.empty() && oImg.cols > 0 && oImg.rows > 0) {
        int copyW = std::min<int>(oImg.cols, tempImg.cols);
        int copyH = std::min<int>(oImg.rows, tempImg.rows);
        oImg.copyTo(tempImg(cv::Rect(0, 0, copyW, copyH)));
    }

    oImg = tempImg;

    return true;
}

// 推論実行
std::optional<std::vector<YOLODetector::BoundingBox>> YOLODetector::inference(const cv::Mat& image)
{
    const int N = 1;   // batch size
    const int C = 3;   // number of channels
    const int W = 1280; // width
    const int H = 736;  // height

    // アスペクト比維持のリサイズとパディング計算
    float targetRatio = static_cast<float>(W) / H;
    float imgRatio = static_cast<float>(image.cols) / image.rows;
    
    cv::Mat resizedImg;
    float padX = 0, padY = 0;  // パディング量
    float scaleX = 1.0f, scaleY = 1.0f;  // スケール倍率

    if (imgRatio >= targetRatio) {
        // 横長：横幅基準でリサイズ（縦はパディング）
        int newWidth = W;
        int newHeight = static_cast<int>(image.rows * W / image.cols);
        cv::resize(image, resizedImg, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        
        // パディング計算（上下に均等に黒塗り）
        padY = (H - newHeight) / 2.0f;
        scaleX = static_cast<float>(W) / image.cols;  // X 軸スケール
        scaleY = static_cast<float>(newHeight) / image.rows;  // Y 軸スケール

    } else {
        // 縦長：縦幅基準でリサイズ（横はパディング）
        int newHeight = H;
        int newWidth = static_cast<int>(image.cols * H / image.rows);
        cv::resize(image, resizedImg, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        
        // パディング計算（左右に均等に黒塗り）
        padX = (W - newWidth) / 2.0f;
        scaleX = static_cast<float>(newWidth) / image.cols;  // X 軸スケール
        scaleY = static_cast<float>(H) / image.rows;  // Y 軸スケール
    }

    // パディング（黒塗り）の適用
    cv::Mat paddedImg(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    resizedImg.copyTo(paddedImg(cv::Rect(static_cast<int>(padX), static_cast<int>(padY), 
                                          resizedImg.cols, resizedImg.rows)));

    // リサイズ倍率を保存（座標逆変換用）
    resizeScales = std::max(scaleX, scaleY);

    // blob 作成
    cv::Mat blob;
    cv::cvtColor(paddedImg, blob, cv::COLOR_BGR2RGB);
    cv::dnn::blobFromImage(blob, blob, 1.0 / 255.0, cv::Size(), 
                           cv::Scalar(0, 0, 0), true, false);

    std::vector<int64_t> input_tensor_shape = { static_cast<int64_t>(N), static_cast<int64_t>(C), 
                                                 static_cast<int64_t>(H), static_cast<int64_t>(W) };
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *m_->memory_info,
        blob.ptr<float>(),
        static_cast<size_t>(blob.total()),
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    // セッション実行
    auto output_tensors = m_->session->Run(Ort::RunOptions{ nullptr }, 
                                           m_->input_node_names.data(), 
                                           &input_tensor, 1, 
                                           m_->output_node_names.data(), 
                                           m_->num_output_nodes);

    std::vector<BoundingBox> bboxes;

    // 出力形状を柔軟に扱う
    auto info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    if (shape.size() != 3) {
        return bboxes;
    }

    int64_t batch = shape[0];
    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];

    int64_t detections = 0;
    int64_t per_det = 0;

    // case: (N, 300, 6) - YOLO2 形式
    if (dim2 == 6) {
        detections = dim1;
        per_det = dim2;
    }
    // case: (N, values, count) - 旧形式
    else if (dim1 == 6) {
        detections = dim2;
        per_det = dim1;
    }
    else {
        // フェールセーフ：既存ロジックに従う
        per_det = static_cast<int>(shape[1]);
        detections = static_cast<int>(shape[2]);
    }

    float const* output_tensor = output_tensors[0].GetTensorData<float>();
    float score_threshold = 0.1f;

    // バッチ 0 を扱う
    const float* batch_base = output_tensor;

    for (int64_t i = 0; i < detections; ++i) {
        const float* det = batch_base + i * per_det;
        
        if (per_det < 6) continue;

        float score = det[4];
        if (score <= score_threshold) continue;

        float x1_model = det[0];
        float y1_model = det[1];
        float x2_model = det[2];
        float y2_model = det[3];
        int class_id = static_cast<int>(det[5]);

        // モデル座標 -> 元画像座標（逆変換）
        // パディングオフセットを差し引き、個別スケールでスケーリング
        float x1 = (x1_model - padX) / scaleX;
        float y1 = (y1_model - padY) / scaleY;
        float x2 = (x2_model - padX) / scaleX;
        float y2 = (y2_model - padY) / scaleY;

         // 左上・幅高さを正しく計算
         float leftVal = (x1 < x2) ? x1 : x2;
         float topVal = (y1 < y2) ? y1 : y2;
         int left = static_cast<int>(std::round(std::max<float>(0.0f, leftVal)));
         int top = static_cast<int>(std::round(std::max<float>(0.0f, topVal)));
        int width = static_cast<int>(std::round(std::abs(x2 - x1)));
        int height = static_cast<int>(std::round(std::abs(y2 - y1)));

        if (width <= 0 || height <= 0) continue;

        BoundingBox bbox;
        bbox.score = score;
        bbox.index = class_id;           // 単純増加インデックス（クラス番号）
        bbox.rect = cv::Rect(left, top, width, height);
        bboxes.push_back(bbox);
    }

    return bboxes;
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
        obj.unseenFrames = 0;            // 未検出フレームカウント
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

// DirectML 初期化（Windows のみ）
bool YOLODetector::initializeDirectML()
{
#ifdef _WIN32
    try {
        m_->session_options_dml = Ort::SessionOptions();
        
        m_->session_options_dml.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_->session_options_dml.SetLogSeverityLevel(3);
        m_->session_options_dml.SetIntraOpNumThreads(1);

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

        // CPU 用メモリ情報も保持（フォールバック用）
        m_->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault));

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
