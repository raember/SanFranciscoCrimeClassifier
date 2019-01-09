#!/bin/env python3

from typing import Tuple

import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.naive_bayes import BernoulliNB

MINUTECOLUMNS = {}
for min_int in range(0, 60):
    MINUTECOLUMNS[min_int] = "m{}".format(min_int)
HOURCOLUMNS = {}
for hour_int in range(0, 24):
    HOURCOLUMNS[hour_int] = "H{}".format(hour_int)
DAYCOLUMNS = {}
for day_int in range(0, 30):
    DAYCOLUMNS[day_int] = "D{}".format(day_int + 1)
MONTHCOLUMNS = {}
for month_int in range(0, 11):
    MONTHCOLUMNS[month_int] = "M{}".format(month_int + 1)
YEARCOLUMNS = {}
for year_int in range(2003, 2015):
    YEARCOLUMNS[year_int] = "Y{}".format(year_int)


def preprocess_dataframe(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    print("Binarize data")
    minute = pd.get_dummies(data.Dates.dt.minute).rename(columns=MINUTECOLUMNS)
    hour = pd.get_dummies(data.Dates.dt.hour).rename(columns=HOURCOLUMNS)
    day = pd.get_dummies(data.Dates.dt.day).rename(columns=DAYCOLUMNS)
    month = pd.get_dummies(data.Dates.dt.month).rename(columns=MONTHCOLUMNS)
    year = pd.get_dummies(data.Dates.dt.year).rename(columns=YEARCOLUMNS)
    weekdays = pd.get_dummies(data.DayOfWeek)
    districts = pd.get_dummies(data.PdDistrict)
    x = data.X
    y = data.Y
    print("Assemble new array")
    new_data = pd.concat([minute, hour, day, month, year, weekdays, districts, x, y], axis=1)
    columns = new_data.keys().tolist()
    return new_data, columns


def evaluate(prediction, labels):
    print("LogLoss: {}".format(log_loss(labels, prediction)))
    predicted_crime = np.argmax(prediction, axis=1)
    print("Accuracy: {}%".format(accuracy_score(labels, predicted_crime) * 100))


print("Load Data with pandas, and parse the first column into datetime")
train = pd.read_csv('train.csv', parse_dates=['Dates'])
test = pd.read_csv('test.csv', parse_dates=['Dates'])

print("Convert crime labels to numbers")
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)

print("Build training data")
train_data, features = preprocess_dataframe(train)
train_data['crime'] = crime

print("Features[{}]: {}".format(len(features), np.array(features)))

print("Split up training data")
# training, validation = train_test_split(train_data, test_size=.20)
training = train_data
validation = train_data

# Bernoulli Naïve Bayes
print("Train Bernoulli Naïve Bayes classifier")
air_bnb = BernoulliNB()
air_bnb.fit(training[features], training['crime'])

print("Predict labels")
predicted = air_bnb.predict_proba(validation[features])

print("Validate prediction")
evaluate(predicted, validation['crime'])

# Predict crimes of test dataset
print("Build test data")
test_data, _ = preprocess_dataframe(test)

print("Predict test labels")
predicted = air_bnb.predict_proba(test_data[features])

print("Write results")
result = pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('testResult.csv', index=True, index_label='Id')

print("Train Keras")
model = keras.Sequential([
    keras.layers.Dense(80, input_dim=len(features), activation='relu'),
    keras.layers.Dense(118, activation='relu'),
    keras.layers.Dense(39, activation='softmax')
])
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
model.fit(training[features], training['crime'], epochs=5, batch_size=1024)

print("Predict labels")
predicted = model.predict_proba(validation[features])

print("Validate prediction")
evaluate(predicted, validation['crime'])

# Logistic Regression
print("Train Logistic Regression for comparison")
lr = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')
lr.fit(training[features], training['crime'])

print("Predict labels")
predicted = np.array(lr.predict_proba(validation[features]))

print("Validate prediction")
evaluate(predicted, validation['crime'])
