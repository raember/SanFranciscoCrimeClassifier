#!/bin/env python3

import tensorflow as tf
from tensorflow import keras

class model:
    def _train(self, train_data, train_labels, test_data=None, test_labels=None):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        print("Constructed model.")
        # optimizer = tf.train.AdamOptimizer()
        # optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Compiled model.")

        model.fit(train_data, train_labels, epochs=5)
        print("Trained model.")

        if test_data is not None and test_labels is not None:
            test_loss, test_acc = model.evaluate(test_data, test_labels)
            print("Tested model.")
            print("Test accuracy: {}".format(test_acc))

        model.save("./model.h5")
        print("Saved model.")
        return model

    @staticmethod
    def get_model(train_data=None, train_labels=None, test_data=None, test_labels=None):
        if train_data is not None and train_labels is not None:
            return model()._train(train_data, train_labels, test_data, test_labels)
        else:
            return keras.models.load_model("./model.h5", compile=False)
