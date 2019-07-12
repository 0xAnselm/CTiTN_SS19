""" Module documentation:
https://medium.com/codingthesmartway-com-blog/the-machine-learning-crash-course-part-2-linear-regression-6a5955792109
__author__ = "R"
__copyright__ = ""
__credits__ = ["Sebastian Eschweiler"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "R"
__email__ = "~"
__status__ = "Production"

TO DO:
Show predictions in Plot
Show Weights
Fit to tensorboard
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

EPOCHS = 200
NAME = "Test_R{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=''.format(NAME))


def linreg_main():
    vec1, vec2 = training_data()
    model = model_cfg()
    print(model.summary())
    trained_model, history = training(model, vec1, vec2)
    plotty(trained_model, vec1, vec2, history)


def plotty(trained_model, vec1, vec2, history):
    # axes = plt.gca()
    # axes.set_xlim([-1,10])
    # axes.set_ylim([-1,30])
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])

    plt.subplot(2, 1, 2)
    x = np.linspace(-10, 20, 100)
    # plt.title("Graph of f(x)=2x+30")
    # plt.plot(x, x * 2 + 30)
    plt.scatter(vec1.numpy(), vec2.numpy(), color="blue")
    # plt.scatter(20,trained_model.predict([20.0]), color="green")
    for i in range(-10, 20):
        plt.scatter(i, trained_model.predict([i]), color="red", s=7)
    plt.show()
    print("The Weights:", trained_model.layers[0].get_weights())


def training_data():
    tf_values_x = tf.Variable([-10, -5, 0, 2, 6, 12, 15, 2], dtype=float)
    tf_values_y = tf.add(tf.multiply(tf_values_x, 2), 30)

    # for i, x in enumerate(values_x):
    #     print("X: {} Y: {}".format(x, values_y[i]))

    return (tf_values_x, tf_values_y)


def model_cfg():
    # A Dense layer can be seen as a linear operation in which every input is connected to every output by a weight and a
    # bias. The number of inputs is specified by the first parameter units. The number of neurons in the layer is
    # determined by the value of the parameter input_shape.
    #         model = tf.keras.Sequential([
    #             tf.keras.layers.Dense(units=1, input_shape=[1])
    #         ])

    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(units=1, input_shape=[1], name="my_DenseLayer"))

    model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))
    return model2


def test():
    print("Hi")


def training(model, vec1, vec2):
    history = model.fit(vec1, vec2, epochs=EPOCHS, verbose=0, callbacks=[tensorboard])
    # plt.xlabel("Epoch Number")
    # plt.ylabel("Loss Magnidute")
    # plt.plot(history.history['loss'])
    # plt.show()

    return (model, history)
