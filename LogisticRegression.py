import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]	 

train_x = pd.read_csv('examples.csv', sep=',',header=None)
train_y = pd.read_csv('classes.csv', sep=',',header=None)
train_y = np.asarray(train_y).flatten()

test_x = pd.read_csv('testing.csv', sep=',',header=None)
#test_y = pd.read_csv('test_y.csv', sep=',',header=None)
#test_y = np.asarray(test_y).flatten()


print("Creating model...")
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)
#print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

predictions = mul_lr.predict(test_x)
testing_ids = np.genfromtxt('testing_ids.csv', dtype=object, delimiter=',')

results = []
i = 0
with open("results.csv", 'w') as f:
    f.write("id,class\n")
    for r in predictions:
        name = "validation." + str(testing_ids[i]) + ".au"
        p = predictions[i]      
        line = name + "," + genres[p] + "\n"
        f.write(line)
        i += 1
        
f.close()