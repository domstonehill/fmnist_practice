import pandas as pd
import numpy as np
from configs import *
import matplotlib.pyplot as plt

# Make numpy easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


def load_data(filepath: str, num_labels: int) -> (np.array, np.array):
    '''
    Loads a csv dataset where one column is named 'label' and contains the class labels, and all the other columns are
    the features and are in the range of [0, 255]. Filepath should be a csv file that contains the data, the num_labels
    should be an integer of the number of labels to be used for one-hot encoding. Returns the features, and labels as
    numpy arrays. Labels are one-hot encoded

    :param filepath: str, path to csv file
    :param num_labels: int, number of classes

    :return: features:np.array, labels:np.array
    '''
    data = pd.read_csv(filepath)

    features = data.copy()
    labels = data.pop('label')

    features = np.array(features) / 255
    labels = tf.one_hot(np.array(labels), num_labels)

    return features, labels


def build_model(num_labels: int):
    '''
    Builds a tf.keras model. Feel free to tinker with the architecture. The number of labels needs to be input so that
    the correct number of output nodes can be created. This model is designed for one-hot encoded labels

    :param num_labels: int, number of categories
    :return: tf.keras compiled model
    '''
    model = tf.keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_labels, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ]
    )

    return model


if __name__ == '__main__':
    # Defining the number of categories, since this is the standard Fashion MNIST, there are 10 labels. It could be
    # more efficient to calculate this from the input data, but there could be situations where some datasets don't have
    # an example for every class
    num_labels = 10

    # Load in the training data
    fmnist_features, fmnist_labels = load_data(
        DATA_DIR + '/fashion-mnist_train.csv',
        num_labels
    )

    # Load validation data, leave as tuple because the validation_data arg in model.fit requires a tuple of (x, y)
    val_data = load_data(
        DATA_DIR + '/fashion-mnist_test.csv',
        num_labels
    )

    # Build and compile the model
    fmnist_model = build_model(num_labels)

    # Train the model, validation data included
    history = fmnist_model.fit(
        fmnist_features,
        fmnist_labels,
        validation_data=val_data,
        epochs=25,
        verbose=1
    )

    # plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(['Train', 'Validation'])

    ax2.plot(history.history['categorical_accuracy'])
    ax2.plot(history.history['val_categorical_accuracy'])
    ax2.set_title('Categorical Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.show()
