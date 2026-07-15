
## WoLNamesBlackedOut for OBS (OBS Plugin for FFXIV)

OBSで配信（キャプチャ）しているFF14の画面からキャラクター名を隠すプラグインです。<br>
DirectMLでユーザー名を検出しフィルタします。<br>
[![Watch the video](https://img.youtube.com/vi/cs6Qgfgf6Is/hqdefault.jpg)](https://www.youtube.com/watch?v=cs6Qgfgf6Is)<br>
オリジナルはこちらです。<br>
https://github.com/royshil/obs-detect

このフォーク版では、YOLODetector クラスを追加し、カスタム YOLO26 モデルのサポートをしています。<br>
また推論をスレッド化することでFPSを改善しています。
BytetrackとPaddleOCRを利用して名前指定でマスク処理の対象外にできます。

### オリジナルからの変更
- YOLODetector クラス: 新規追加の YOLO26 物体検出クラス
- ONNX Runtimeの推論をスレッド化対応し当方環境で50FPSを実現
- EdgeYOLO、顔検出、トラッキングなど削除
- マスク対象外エリアを設定可能とし、簡易的ですが配信者のキャラクターをマスク対象外にできるようにしました。
- ByteTrack、PaddleOcrで名前でマスク対象外ができるようになりました。
- OBS-DX11-DX12-DirectMLと連携しGPU内で完結することで当方環境で60FPSを実現しました。

[![Watch the video](https://img.youtube.com/vi/-JE25IVAbdI/hqdefault.jpg)](https://www.youtube.com/watch?v=-JE25IVAbdI)<br>

## 動作環境
以下で動作確認しています。

- Windows11 25H2 64bit
- OBS 32.1.2
- Intel 11400F
- Geforce RTX 4700TiS 16GB<br>
DirectMLを利用しているためRadeonでも動作すると思われます。<br>

FF14を実行しながらのYOLO26、PaddleOCRのDirectML推論になりますが、負荷もそこまで大きくなく、ゲームプレイに支障はありません。<br>
下記の動画では、FF14で名前を消しながらタスクマネージャーを表示して負荷を確認しています。<br>
[![Watch the video](https://img.youtube.com/vi/Rh45E43iMpE/hqdefault.jpg)](https://www.youtube.com/watch?v=Rh45E43iMpE)<br>

## インストール
1. [リリース](https://github.com/calocenrieti/obs-wolnamesblackedout/releases)から最新のobs-wolnamesblackedout_x.x.x.zipをダウンロードします。
2. OBSを終了します。
3. ZIPを全て展開し、出てくるobs-wolnamesblackedoutフォルダを%ProgramData%\obs-studio\plugins\にコピーします。<br>
もしくは、同梱しているinstall.ps1の内容を確認の上、を右クリックし"Powershellで実行する"を選択ください。上記フォルダにコピーします。管理者権限が必要なためsmart screenの確認画面がでます<br>
（通常`C:\ProgramData\obs-studio\plugins\`です。<br>
ProgramDataは隠しフォルダになっているので、エクスプローラーのアドレスバーにコピペし、該当フォルダを開くのがおすすめです。）
4. OBSでゲームキャプチャにフィルタ”WoLNamesBlackedOut”を追加して利用します。<br>

設定など詳細は下記サイトを確認ください。<br>
https://blog.calocenrieti.com/blog/obs-wolnamesblackedout/

## License
This project is licensed under the GPL v2 (or later).

This plugin is based on the open-source project
[obs-detect](https://github.com/royshil/obs-detect) (GPL v2).
Copyright (c) Roy Shilkrot.

This plugin also references implementation ideas from
[winter1l/obs-detect](https://github.com/winter1l/obs-detect) (GPL v2),
specifically for GPU zero-copy processing. No direct source code was used.

The distribution includes third-party libraries under their respective licenses.
See the LICENSES folder for details.

## Third Party Libraries & Licenses

This project incorporates the following third-party components:

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** (MIT): High-performance ML inference runtime
- **[DirectML](https://github.com/microsoft/DirectML)** (MIT): Hardware-accelerated DirectX 12 library for ML
- **[OpenCV](https://github.com/opencv/opencv)** (Apache 2.0): Computer vision library

- **[Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)** (AGPL-3.0): Real-time object detection model
  This project utilizes a customized YOLO26 model exported to ONNX.
  *Note: Only the exported ONNX model is used. This project itself is not licensed under AGPL.*

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Apache 2.0): Lightweight OCR toolkit
  This project utilizes the en_PP-OCRv5_mobile_rec recognition model exported to ONNX.

- **[ByteTrack-cpp](https://github.com/derpda/ByteTrack-cpp)** (MIT): C++ implementation of ByteTrack

- **[Eigen](https://gitlab.com/libeigen/eigen)** (MPL 2.0): C++ template library for linear algebra
