#!/bin/env python3

import tensorflow as tf
# from tensorflow import keras
import keras
import logging as log
import numpy
from numpy import ndarray


class Model:
    file = "model.h5"
    log = None

    def __init__(self):
        self.log = log.getLogger(self.__class__.__name__)

    def _train(self, train_data, train_labels, test_data=None, test_labels=None):
        r"""Construct and train model

        :param train_data: The training data
        :type train_data: ndarray
        :param train_labels: The training labels
        :type train_labels: ndarray
        :param test_data: The test data for verification
        :type test_data: ndarray
        :param test_labels: The test labels for verification
        :type test_labels: ndarray
        :return:
        """
        model = keras.Sequential([
            keras.layers.Dense(16, input_shape=(6,), activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(40, activation='softmax')
            # keras.layers.Flatten(input_shape=(28, 28)),
            # keras.layers.Dense(128, activation=tf.nn.relu),
            # keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.log.info("Constructed model")
        # optimizer = tf.train.AdamOptimizer()
        # optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        optimizer = keras.optimizers.Adam(lr=0.04)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.log.info("Compiled model")

        # Delete data to make the training faster
        # train_data = numpy.delete(train_data, numpy.s_[100::], axis=0)
        # train_labels = numpy.delete(train_labels, numpy.s_[100::], axis=0)

        # Train model
        model.fit(train_data, train_labels, epochs=5, batch_size=20)
        self.log.info("Trained model")

        if test_data is not None and test_labels is not None:
            test_loss, test_acc = model.evaluate(test_data, test_labels)
            self.log.info("Tested model")
            self.log.info("Test accuracy: {}".format(test_acc))

        model.save(self.file)
        self.log.info("Saved model")
        return model

    def get_model(self, train_data=None, train_labels=None, test_data=None, test_labels=None):
        if train_data is not None and train_labels is not None:
            return self._train(train_data, train_labels, test_data, test_labels)
        else:
            self.log.debug("Loading model")
            return keras.models.load_model(self.file, compile=False)
