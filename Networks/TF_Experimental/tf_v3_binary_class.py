import tensorflow as tf
import numpy as np

EPOCHS = 20


def data():
    x_test = tf.Variable(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0],
         [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    x_len = x_test.shape[0]
    x_test = tf.reshape(x_test, [x_len, 1, 4])

    y_test = tf.Variable(
        [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]])
    y_len = y_test.shape[0]
    y_test = tf.reshape(y_test, [y_len, 1, 2])
    train = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    x_val = tf.Variable([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 1]])
    x_len = x_val.shape[0]
    x_val = tf.reshape(x_val, [x_len, 1, 4])

    y_val = tf.Variable([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_len = y_val.shape[0]
    y_val = tf.reshape(y_val, [y_len, 1, 2])

    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    return train, val


def model(train, val):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=4, input_shape=(4,)))
    #model.add(tf.keras.layers.Dense(units=10))
    model.add(tf.keras.layers.Dense(units=2))

    model.compile(loss="mean_squared_error", optimizer="sgd")
    model.fit(train, epochs=EPOCHS, verbose=2)

    a = model.predict(val)
    for k, v in a:
        print([k, v])

    for k, v in val:
        print(v.numpy())
    return model


def run():
    train, val = data()
    model(train, val)
