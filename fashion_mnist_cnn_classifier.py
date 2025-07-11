# ディープラーニングのモデルを構築する
# データセットはFashion MNIST
# モデルは畳み込みニューラルネットワークを使用

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam


def plot_history(history):
    """学習履歴（損失と正解率）をグラフ表示する"""
    plt.figure(figsize=(12, 5))

    # 損失
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train", color="black")
    plt.plot(history.history["val_loss"], label="val", color="red")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()

    # 正解率
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train", color="black")
    plt.plot(history.history["val_accuracy"], label="val", color="red")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    # ===================== データ準備 =====================
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 形状変換 & 正規化
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # ===================== モデル構築 =====================
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()

    # ===================== 学習 =====================
    batch_size = 32
    epochs = 5
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )

    elapsed = time.time() - start_time

    # ===================== 評価 =====================
    loss, accuracy = model.evaluate(x_test, y_test, verbose="silent")
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Training time: {elapsed:.2f} sec\n")

    # ===================== グラフ描画 =====================
    plot_history(history)


if __name__ == "__main__":
    main()
    