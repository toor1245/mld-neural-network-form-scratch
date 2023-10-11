import numpy as np
import tensorflow as tf
from keras.src.optimizers import SGD
from tensorflow.python.keras.utils import np_utils

import keras
from keras.datasets import mnist
from keras import layers


def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

sgd_optimizer = SGD(learning_rate=0.1)

model = keras.Sequential([
    layers.Dense(40, input_shape=(28 * 28,)),
    layers.Activation(tf.nn.tanh),
    layers.Dense(10, input_shape=(40,)),
    layers.Activation(tf.nn.tanh),
])

model.compile(optimizer=sgd_optimizer, loss='mean_squared_error')

model.fit(x_train, y_train, epochs=100)

test_digits = x_test
prediction = model.predict(test_digits)
print(prediction)

for i in range(len(prediction)):
    print(
        f"true value: {np.argmax(y_test[i])}, pred idx: {np.argmax(prediction[i])}, value: {prediction[i, np.argmax(prediction[i])]}")
    print("\n")
