import keras.layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Reshape
from keras.activations import sigmoid
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
from tensorflow.python.keras.utils import np_utils
from keras.datasets import mnist


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 28, 28, 1)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

model = Sequential([
    Conv2D(5, kernel_size=(3, 3), input_shape=(28, 28, 1)),
    Activation(sigmoid),
    Reshape((5, 26, 26)),
    Flatten(),
    Dense(100),
    Activation(sigmoid),
    Dense(2),
    Activation(sigmoid),
])

model.compile(loss=binary_crossentropy, optimizer=SGD())
model.fit(x_train, y_train, epochs=20, batch_size=1)

test_digits = x_test
prediction = model.predict(test_digits)

for i in range(len(prediction)):
    print(
        f"true value: {np.argmax(y_test[i])}, pred idx: {np.argmax(prediction[i])}, value: {prediction[i, np.argmax(prediction[i])]}")
