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

EPOCHS = 500
ETA = 0.1
NAME = "LinReg_Example{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='LinReg_Example'.format(NAME))
baselogger = tf.keras.callbacks.BaseLogger(stateful_metrics=None)
progbar = tf.keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)

def linreg_example_main():
    vec_x, vec_y = training_data()
    model = model_cfg()
    print(model.summary())
    trained_model, history = training(model, vec_x, vec_y)
    plotty(trained_model, vec_x, vec_y, history)


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
    for i in range(150, 170):
        plt.scatter(i, trained_model.predict([i]), color="red", s=7)
    plt.show()
    print("The Weights:", trained_model.layers[0].get_weights())
    print("Prediction:", trained_model.predict([158.9]))


def training_data():
    # Female Bodysizes
    tf_values_x = tf.Variable([156.3, 158.9, 160.8, 179.6, 156.6, 165.1, 165.9, 156.7, 167.8, 160.8], dtype=float, name="W")
    # Female Ringfingersizes
    tf_values_y = tf.Variable([47.1, 46.8, 49.3, 53.2, 47.7, 49.0, 50.6, 47.1, 51.7, 47.8], dtype=float, name="b")

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
    model2.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

    model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(ETA))
    return model2


def training(model, vec_x, vec_y):
    print("Model weights:", model.layers[0].get_weights())
    history = model.fit(x=vec_x, y=vec_y, epochs=EPOCHS, verbose=2, batch_size=20, callbacks=[tensorboard, baselogger, progbar])
    # plt.xlabel("Epoch Number")
    # plt.ylabel("Loss Magnidute")
    # plt.plot(history.history['loss'])
    # plt.show()

    return (model, history)
