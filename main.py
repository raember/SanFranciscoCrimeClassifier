#!/bin/env python3

import argparse
import logging

import tensorflow as tf
from tensorflow import keras

from model import Model
from preprocessor import *

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

trainsamplesfile = TrainDataCsvFile()
trainlabelfile = TrainLabelsCsvFile(trainsamplesfile)
testsamplesfile = TestDataCsvFile()
for file in [trainsamplesfile, trainlabelfile, testsamplesfile]:
    if args.prep_data or not file.prep_file_exists():
        file.parse()
        file.save()
    else:
        file.load()
# for key in trainlabelfile.stats:
#     trainlabelfile.stats[key] /= trainlabelfile.count
#     print("{} : {}%".format(trainlabelfile.stats[key] * 100, trainlabelfile.CATEGORIES[key]))
# exit(0)
# print(trainsamplesfile.df.columns.values)
# print(trainlabelfile.df.columns.values)
# exit(0)
# print(trainsamplesfile.get(int(0)))
# exit(0)
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
    # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    mdl = Model().get_model(
        # scaler.fit_transform(trainsamplesfile.toNpArray()).reshape(trainsamplesfile.df.shape),
        trainsamplesfile.toNpArray(),
        trainlabelfile.toNpArray()
    )
else:
    mdl = Model().get_model()

# predictions = mdl.predict(testfile.toNpArray())
predictions = mdl.predict(trainsamplesfile.toNpArray())
for i in range(0, 19):
    predicted = trainlabelfile.CATEGORIES[np.argmax(predictions[i])]
    actual = trainlabelfile.get(i)
    logging.info("{} ?= {}".format(actual, predicted))
