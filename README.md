
## OBS-WoLNamesBlackedOut (OBS Plugin for FFXIV)

OBSで配信（キャプチャ）しているFF14の画面からキャラクター名を隠すプラグインです。<br>
DirectMLでユーザー名を検出しフィルタします。

オリジナルはこちらです。
https://github.com/royshil/obs-detect

https://github.com/royshil/obs-detect


このフォーク版では、YOLODetector クラスを追加し、カスタム YOLO26 モデルのサポートをしています。

### オリジナルからの変更
- **YOLODetector クラス**: 新規追加の YOLO 物体検出クラス
  - ONNX Runtime を使用したカスタム YOLO26 モデル対応
  - インデックスベースのクラス名付け (0, 1, 2, ...)
  - 自動生成クラス名："class_0", "class_1" など
- EdgeYOLO、顔検出、トラッキングなど削除

## インストール
1. [リリース](https://github.com/calocenrieti/obs-wolnamesblackedout/releases)から最新のobs-wolnamesblackedout_x.x.x.zipをダウンロードします。
2. OBSを終了します。
3. ZIPを解凍して出てくる`obs-wolnamesblackedout`フォルダを`%ProgramData%\obs-studio\plugins\`にコピーします。<br>
（通常`C:\ProgramData\obs-studio\plugins\`です。<br>
エクスプローラーのアドレスバーにコピペし、該当フォルダを開くのがおすすめです。）
4. OBSでゲームキャプチャにフィルタ”DETECT”を追加して利用します。

## Third Party Libraries & Licenses

This project incorporates the following third-party components:

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** (MIT): High-performance ML inference runtime
- **[OpenCV](https://github.com/opencv/opencv)** (Apache 2.0): Computer vision library

**Note**: This project is distributed under the GPLv2 license as per the original work.

