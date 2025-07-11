# TensorFlowとKerasによる画像分類チュートリアル

このリポジトリは、TensorFlowとKerasを用いた画像分類の基本的な実装例をまとめたものです。  
各スクリプトは異なるデータセットやモデル構造を扱っており、画像認識におけるディープラーニングの基礎を学ぶ教材として活用できます。

---

## 内容一覧

| ファイル名                          | データセット   | モデル種別        | 概要                                 |
|-----------------------------------|----------------|-------------------|--------------------------------------|
| `mnist_digit_dense_classifier.py` | MNIST          | 全結合（Dense）   | 手書き数字分類の基本モデル           |
| `fashion_mnist_dense_classifier.py` | Fashion-MNIST  | 全結合（Dense）   | 衣料画像分類（Denseネット）         |
| `fashion_mnist_cnn_classifier.py`   | Fashion-MNIST  | CNN               | 畳み込みネットワークによる高精度分類 |
| `cifar10_image_classifier.py`       | CIFAR-10       | 深層CNN            | カラー画像分類（より複雑な構成）    |

---

## 特徴

- 各ファイルは独立して実行可能
- 学習過程と評価指標（損失・正解率）をグラフで可視化
- `plot_history()` 関数で可視化部分を共通化
- モデル構造は `model.summary()` により確認可能

---

## 必要なライブラリ

以下のコマンドで依存ライブラリを一括インストールできます：

```bash
pip install tensorflow keras numpy matplotlib
