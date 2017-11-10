import pandas as pd
from sklearn import linear_model
from sklearn import metrics


train_x = pd.read_csv('examples.csv', sep=',',header=None)
train_y = pd.read_csv('classes.csv', sep=',',header=None)
print(train_x.shape)
print(train_y.shape)


print("Creating model...")
#mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)
#print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
#print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
