"""
https://stackoverflow.com/questions/52736517/tensorflow-keras-with-tf-dataset-input
"""

import pandas as pd
import tensorflow as tf

ETA = 0.1
EPOCH = 5
LOG_DIR = "V2_BinClass"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_grads=1)


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=4, activation="sigmoid", input_shape=(10,), kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Ones(), name="Input_Layer"))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Ones(), name="Output_Layer"))
    sgd = tf.keras.optimizers.SGD(lr=ETA)
    adam = tf.keras.optimizers.Adam(lr=ETA)
    model.compile(optimizer=adam, loss="mean_squared_error", metrics=["accuracy"])
    return model


def data():
    with open("C:/Users/Robert/Dropbox/Robert/Seminar/CITN_SS19/Networks/TF_Experimental/train_data.csv",
              newline="") as f:
        reader = pd.read_csv(f, delimiter=",", usecols=['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', 'Class'],
                             skip_blank_lines=True)
    features = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
    labels = ['Class']
    training_df: pd.DataFrame = reader
    # print(training_df)
    x = tf.cast(training_df[features].values, tf.int32)
    x_len = x.shape[0]
    x = tf.reshape(x, [x_len, 1, 10])

    y = tf.cast(training_df[labels].values, tf.int32)
    y_len = y.shape[0]
    y = tf.reshape(y, [y_len, 1])
    dataset = (tf.data.Dataset.from_tensor_slices((x,y)))

    # Dataset.shard() is also an option
    dataset_size = len(list(dataset))
    train_size = int(0.7 * dataset_size)
    test_size = int(0.15 * dataset_size)
    val_size = int(0.15 * dataset_size)
    dataset = dataset.shuffle(buffer_size=1000)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset, test_dataset, val_dataset


def run_model(model, train, test):
    history = model.fit(train, epochs=EPOCH, validation_data=test, callbacks=[tensorboard_callback], shuffle=True)
    return model, history


def plot_history(history):
    hist = pd.DataFrame(history.history)
    print(hist)


def bin_classifier_main():
    train, test, val = data()
    model = create_model()
    run_model(model, train, test)
