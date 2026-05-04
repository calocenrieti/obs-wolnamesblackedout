
## WoLNamesBlackedOut for OBS (OBS Plugin for FFXIV)

OBSで配信（キャプチャ）しているFF14の画面からキャラクター名を隠すプラグインです。<br>
DirectMLでユーザー名を検出しフィルタします。

オリジナルはこちらです。
https://github.com/royshil/obs-detect

このフォーク版では、YOLODetector クラスを追加し、カスタム YOLO26 モデルのサポートをしています。<br>
また推論をスレッド化することでFPSを改善しています。

### オリジナルからの変更
- YOLODetector クラス: 新規追加の YOLO26 物体検出クラス
- ONNX Runtimeの推論をスレッド化対応し当方環境で50FPSを実現
- EdgeYOLO、顔検出、トラッキングなど削除
- マスク対象外エリアを設定可能とし、簡易的ですが配信者のキャラクターをマスク対象外にできるようにしました。

## 動作環境
以下で動作確認しています。

- Windows11 25H2 64bit
- OBS 32.1.2
- Intel 11400F
- Geforce RTX 4700TiS 16GB<br>
DirectMLを利用しているためRadeonでも動作すると思われます。<br>

FF14を実行しながらのDirectML推論になりますが、負荷もそこまで大きくなく、ゲームプレイに支障はありません。<br>
下記の動画では、FF14で名前を消しながらタスクマネージャーを表示して負荷を確認しています。<br>
https://www.youtube.com/live/Rh45E43iMpE?si=8hvcouXaAgVLhm3i&t=3317

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

## Third Party Libraries & Licenses

This project incorporates the following third-party components:

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** (MIT): High-performance ML inference runtime
- **[OpenCV](https://github.com/opencv/opencv)** (Apache 2.0): Computer vision library

**Note**: This project is distributed under the GPLv2 license as per the original work.

