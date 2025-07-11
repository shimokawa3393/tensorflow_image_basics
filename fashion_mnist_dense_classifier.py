# ディープラーニングのモデルを構築する
# データセットはFashion MNIST
# モデルは全結合層を使用

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # ===================== モデル構築 =====================
    model = Sequential()
    model.add(Dense(512, input_shape=(784,), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()

    # ===================== 学習 =====================
    batch_size = 32
    epochs = 10
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
