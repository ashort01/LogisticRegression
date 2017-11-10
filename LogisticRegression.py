import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics


train_x = np.genfromtxt('examples.csv', delimiter=',') 
train_y = np.genfromtxt('classes.csv', delimiter=',') 
train_y = train_y.flatten()

test_x = np.genfromtxt('test_x.csv', delimiter=',') 
test_y = np.genfromtxt('test_y.csv', delimiter=',') 
test_y = test_y.flatten()


print("Creating model...")
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
