#!/bin/env python3

import argparse
import logging

import tensorflow as tf
from sklearn.metrics import log_loss, accuracy_score
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

if args.train:
    mdl = Model().get_model(
        trainsamplesfile.toNpArray(),
        trainlabelfile.toNpArray()
    )
else:
    mdl = Model().get_model()

predictions = mdl.predict(trainsamplesfile.toNpArray())
print("LogLoss: {}".format(log_loss(trainlabelfile.toNpArray(), predictions)))
predicted_crime = np.argmax(predictions, axis=1)
print("Accuracy: {}%".format(accuracy_score(trainlabelfile.toNpArray(), predicted_crime) * 100))

for i in range(0, 19):
    predicted = trainlabelfile.CATEGORIES[np.argmax(predictions[i])]
    actual = trainlabelfile.get(i)
    logging.info("{} ?= {}".format(actual, predicted))

# for key in trainlabelfile.stats:
#     trainlabelfile.stats[key] /= trainlabelfile.count
#     print("{} : {}%".format(trainlabelfile.stats[key] * 100, trainlabelfile.CATEGORIES[key]))

# for i in range(0, 10):
#     # (date, _, _, address, lat, long) = data.get(i)
#     print(trainfile.get(i), trainlabelfile.get(i))
#     webbrowser.open(
#         "https://www.google.ch/maps/"
#         "@{},{},58m/data=!3m1!1e3".format(lat, long)
#     )
