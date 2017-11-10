import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics

train_x = pd.read_csv('examples.csv', sep=',',header=None)
train_y = pd.read_csv('classes.csv', sep=',',header=None)
train_y = np.asarray(train_y).flatten()

test_x = pd.read_csv('test_x.csv', sep=',',header=None)
test_y = pd.read_csv('test_y.csv', sep=',',header=None)
test_y = np.asarray(test_y).flatten()

print train_x.shape
print train_y.shape

print("Creating model...")
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
