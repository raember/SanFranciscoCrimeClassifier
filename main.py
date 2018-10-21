#!/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import argparse
from preprocessor import *
import webbrowser
import logging
import sklearn.preprocessing
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)20s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

logging.info("Tensorflow: {}".format(tf.__version__))
logging.info("Keras: {}".format(keras.__version__))
logging.info("Numpy: {}".format(np.version.full_version))

parser = argparse.ArgumentParser(description="Classifies crimes of San Francisco")
parser.add_argument('-t', '--train', action='store_true',
                    help="Create and train the model(default: load model from disk).")
parser.add_argument('-p', '--prep-data', action='store_true',
                    help="Preprocess data files even if the preprocessed data already exists.")
args = parser.parse_args()
if args.prep_data:
    logging.debug("Preparing data from csv files")
if args.train:
    logging.debug("Training model")

trainfile = TrainDataCsvFile()
trainlabelfile = TrainLabelsCsvFile(trainfile)
testfile = TestDataCsvFile()
for file in [trainfile, trainlabelfile, testfile]:
    if args.prep_data or not file.prep_file_exists():
        file.parse()
        file.save()
    else:
        file.load()

# for i in range(0, 10):
#     # (date, _, _, address, lat, long) = data.get(i)
#     print(trainfile.get(i), trainlabelfile.get(i))
#     # webbrowser.open("https://www.google.ch/maps/@{},{},58m/data=!3m1!1e3".format(lat, long))
# exit(0)
#
# def load_prediction_data():
#     return np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
# pred = load_prediction_data()

if args.train:
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    mdl = Model().get_model(
        scaler.fit_transform(trainfile.toNpArray()).reshape(trainfile.df.shape),
        trainlabelfile.toNpArray()
    )
else:
    mdl = Model().get_model()

# predictions = mdl.predict(testfile.toNpArray())
predictions = mdl.predict(trainfile.toNpArray())
for i in range(0, 19):
    predicted = trainlabelfile.CATEGORIES[np.argmax(predictions[i])]
    actual = trainlabelfile.get(i)
    logging.info("{} ?= {}".format(actual, predicted))