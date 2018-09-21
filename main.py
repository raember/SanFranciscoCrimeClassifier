#!/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import model
import argparse
from preprocessor import *
import webbrowser
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)-15s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

parser = argparse.ArgumentParser(description="Classifies crimes of San Francisco")
parser.add_argument('-t', '--train', action='store_true',
                    help="Create and train the model(default: load model from disk).")
parser.add_argument('-p', '--prep-data', action='store_true',
                    help="Preprocess data files even if the preprocessed data already exists.")
args = parser.parse_args()

testdata = TestDataFile()
if args.prep_data or not testdata.prep_file_exists():
    testdata.parse()
    testdata.save()
else:
    testdata.load()

for i in range(0, 10):
    (date, _, _, address, lat, long) = testdata.get(i)
    print(address, date)
    # webbrowser.open("https://www.google.ch/maps/@{},{},58m/data=!3m1!1e3".format(lat, long))
exit(0)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0


def load_prediction_data():
    return np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
pred = load_prediction_data()

if args.train:
    mdl = model.get_model(train_images, train_labels, test_images, test_labels)
else:
    mdl = model.get_model()

predictions = mdl.predict(test_images)
predicted = np.argmax(predictions[0])
print(class_names[predicted])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()