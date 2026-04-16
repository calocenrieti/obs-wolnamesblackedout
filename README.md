
## OBS-WoLNamesBlackedOut (OBS Plugin for FFXIV)

OBSで配信（キャプチャ）しているFF14の画面からキャラクター名を隠すプラグインです。<br>
DirectMLでユーザー名を検出しフィルタします。

オリジナルはこちらです。
https://github.com/royshil/obs-detect

このフォーク版では、YOLODetector クラスを追加し、カスタム YOLO26 モデルのサポートをしています。<br>
また推論をスレッド化することでFPSを改善しています。

### オリジナルからの変更
- YOLODetector クラス: 新規追加の YOLO26 物体検出クラス
- ONNX Runtimeの推論をスレッド化し当方環境で50FPSを実現
- EdgeYOLO、顔検出、トラッキングなど削除

## 動作環境
以下で動作確認しています。

- Windows11 25H2 64bit
- OBS 32.1.1
- Intel 11400F
- Geforce RTX 4700TiS 16GB<br>
DirectMLを利用しているためRadeonでも動作すると思われます。

## インストール
1. [リリース](https://github.com/calocenrieti/obs-wolnamesblackedout/releases)から最新のobs-wolnamesblackedout_x.x.x.zipをダウンロードします。
2. OBSを終了します。
3. ZIPを解凍して出てくる`obs-wolnamesblackedout`フォルダを`%ProgramData%\obs-studio\plugins\`にコピーします。<br>
（通常`C:\ProgramData\obs-studio\plugins\`です。<br>
エクスプローラーのアドレスバーにコピペし、該当フォルダを開くのがおすすめです。）
4. OBSでゲームキャプチャにフィルタ”WoLNamesBlackedOut”を追加して利用します。

## Third Party Libraries & Licenses

This project incorporates the following third-party components:

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** (MIT): High-performance ML inference runtime
- **[OpenCV](https://github.com/opencv/opencv)** (Apache 2.0): Computer vision library

**Note**: This project is distributed under the GPLv2 license as per the original work.

