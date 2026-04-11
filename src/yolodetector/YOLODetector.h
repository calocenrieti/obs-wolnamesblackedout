#ifndef _YOLO_DETECTOR_H
#define _YOLO_DETECTOR_H

#ifdef _WIN32
#include <windows.h>
#include <dml_provider_factory.h>
#endif // _WIN32

#include <obs-module.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <string>

// 既存の Object 構造体を使用（ort-model/types.hpp から）
#include "ort-model/types.hpp"

/**
 * @brief YOLO 物体検出クラス
 * 
 * ONNX Runtime を使用して YOLO モデルによる物体検出を実行します。
 * DirectML または CPU で推論を行います（DirectML が利用できない場合は CPU にフォールバック）。
 */
class YOLODetector {
public:
    /**
     * @brief 境界ボックス構造体
     */
    struct BoundingBox {
        int index;            // クラス ID（単純増加インデックス）
        float score;          // 信頼度スコア
        cv::Rect rect;        // 検出領域（OpenCV Rect）
    };

public:
    __stdcall YOLODetector();
    __stdcall ~YOLODetector();
    
    /**
     * @brief クラスの初期状態チェック
     */
    operator bool() const;
    
#ifdef _WIN32
    /**
     * @brief モデルの読み込み（Windows 用）
     * @param model_path ONNX モデルファイルパス（UTF-16）
     * @return 成功時 true、失敗時 false
     */
    bool __stdcall loadModel(const wchar_t* model_path);
#else
    /**
     * @brief モデルの読み込み（Linux/macOS 用）
     * @param model_path ONNX モデルファイルパス（UTF-8）
     * @return 成功時 true、失敗時 false
     */
    bool __stdcall loadModel(const char* model_path);
#endif // _WIN32
    
    /**
     * @brief 推論実行
     * @param image 入力画像（BGR）
     * @return 検出された境界ボックスのリスト。空の場合は std::nullopt
     */
    std::optional<std::vector<BoundingBox>> __stdcall inference(const cv::Mat& image);
    
    /**
     * @brief 前処理（リサイズ、正規化など）
     * @param iImg 入力画像
     * @param targetWidth 目標幅
     * @param targetHeight 目標高さ
     * @param oImg 出力画像
     * @return 成功時 true
     */
    bool __stdcall PreProcess2(const cv::Mat& iImg, int targetWidth, int targetHeight, cv::Mat& oImg);
    
    /**
     * @brief 推論結果を Object 構造体に変換
     * @param bboxes YOLODetector の BoundingBox ベクトル
     * @return 変換された Object ベクトル（Object 構造体は既存のプロジェクトで使用されている形式）
     */
    static std::vector<Object> __stdcall convertToObjects(const std::vector<BoundingBox>& bboxes);

public:
    /**
     * @brief 推論デバイスの設定
     * @param useGPU GPU を使用する場合 true
     */
    void setUseGPU(bool useGPU);

    /**
     * @brief DirectML の初期化（Windows のみ）
     * @return 成功時 true、失敗時 false
     */
    bool initializeDirectML();

private:
    struct Private;
    std::unique_ptr<Private> m_;
    std::mutex mutex_;
    
public:
    float resizeScales;  // リサイズスケール（座標変換用）
};

#endif // _YOLO_DETECTOR_H
