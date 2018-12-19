import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score
from sklearn.naive_bayes import BernoulliNB

print("Load Data with pandas, and parse the first column into datetime")
train = pd.read_csv('train.csv', parse_dates=['Dates'])
test = pd.read_csv('test.csv', parse_dates=['Dates'])

HOURCOLUMNS = {}
for hour_int in range(1, 24 + 1):
    HOURCOLUMNS[hour_int - 1] = "H{}".format(hour_int)
DAYCOLUMNS = {}
for day_int in range(1, 31 + 1):
    DAYCOLUMNS[day_int] = "D{}".format(day_int)
MONTHCOLUMNS = {}
for month_int in range(1, 12 + 1):
    MONTHCOLUMNS[month_int] = "M{}".format(month_int)

############################################################

print("Convert crime labels to numbers")
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)

print("Get binarized weekdays, districts, and hours")
weekdays = pd.get_dummies(train.DayOfWeek)
districts = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour).rename(columns=HOURCOLUMNS)
day = pd.get_dummies(train.Dates.dt.day).rename(columns=DAYCOLUMNS)
month = pd.get_dummies(train.Dates.dt.month).rename(columns=MONTHCOLUMNS)
x = train.X
y = train.Y

print("Build new array")
train_data: pd.DataFrame = pd.concat([hour, weekdays, day, month, districts, x, y], axis=1)
print(train_data.keys())
# exit(0)
train_data['crime'] = crime

print("Repeat for test data")
weekdays = pd.get_dummies(test.DayOfWeek)
districts = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour).rename(columns=HOURCOLUMNS)
day = pd.get_dummies(test.Dates.dt.day).rename(columns=DAYCOLUMNS)
month = pd.get_dummies(test.Dates.dt.month).rename(columns=MONTHCOLUMNS)

test_data: pd.DataFrame = pd.concat([hour, weekdays, day, month, districts, x, y], axis=1)

# training, validation = train_test_split(train_data, train_size=.60)
training = train_data
validation = train_data

############################################################
print("Bernoulli 1")

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
            'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
            'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
features += HOURCOLUMNS.values()
features += DAYCOLUMNS.values()
features += MONTHCOLUMNS.values()

model = BernoulliNB()
model.fit(training[features], training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
print(log_loss(validation['crime'], predicted))
predicted_crime = np.argmax(predicted, axis=1)
print(accuracy_score(validation['crime'], predicted_crime))

# print("Logistic Regression for comparison")
# model = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')
# model.fit(training[features], training['crime'])
# predicted = np.array(model.predict_proba(validation[features]))
# print(log_loss(validation['crime'], predicted))
# predicted_crime = np.argmax(predicted, axis=1)
# print(accuracy_score(validation['crime'], predicted_crime))

############################################################
# print("Bernoulli 2")
#
# model = BernoulliNB()
# model.fit(train_data[features], train_data['crime'])
# predicted = model.predict_proba(test_data[features])
#
# print("Write results")
# result = pd.DataFrame(predicted, columns=le_crime.classes_)
# result.to_csv('testResult.csv', index=True, index_label='Id')

# features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
#             'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
#             'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

# features2 = [x for x in range(0, 24)]
# features = features + features2
# print(features)
