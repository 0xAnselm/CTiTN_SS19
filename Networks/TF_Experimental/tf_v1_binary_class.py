import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

EPOCHS = 2
ETA = 0.01
NAME = "Binary_Classifier{}".format(int(time.time()))
INPUT_SHAPE = (1, 10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='{}'.format(NAME))
baselogger = tf.keras.callbacks.BaseLogger(stateful_metrics=None)
progbar = tf.keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)


def bin_classifier_main():
    dataset = train_data()
    model = model_cfg()
    history, trained_model = training(model, dataset)
    print(model.summary())


def train_data():
    with open("C:/Users/Robert/Dropbox/Robert/Seminar/CITN_SS19/Networks/TF_Experimental/train_data.csv",
              newline="") as f:
        reader = pd.read_csv(f, delimiter=",", usecols=['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', 'Class'])
    features = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
    labels = ['Class']
    training_df: pd.DataFrame = reader
    # print(training_df)
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(training_df[features].values, tf.float32),
                tf.cast(training_df[labels].values, tf.int32)
            )
        )
    )
    return training_dataset


def model_cfg():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=3, activation="sigmoid", input_shape=(1,), name="First"))
    model.add(keras.layers.Dense(units=4, activation="sigmoid"))
    model.add(keras.layers.Dense(units=3, activation="sigmoid"))
    sgd = keras.optimizers.SGD(lr=ETA, nesterov=False)
    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])
    return model


def training(model, training_dataset):
    dataset_iter = training_dataset.__iter__()
    features, labels = dataset_iter.next()
    history = model.fit(features, labels, epochs=EPOCHS, callbacks=[tensorboard], steps_per_epoch=10)
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    for layer in model.layers:
        print(layer.get_weights())
    return history, model
