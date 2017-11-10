import pandas as pd
import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as skp

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]	 

train_x = pd.read_csv('examples.csv', sep=',',header=None)
train_y = pd.read_csv('classes.csv', sep=',',header=None)
train_y = np.asarray(train_y).flatten()

test_x = pd.read_csv('testing.csv', sep=',',header=None)
#test_y = pd.read_csv('test_y.csv', sep=',',header=None)
#test_y = np.asarray(test_y).flatten()


# this normalization got us 47%
# scaler= StandardScaler()
# train_x = scaler.fit_transform(train_x)
# test_x = scaler.transform(test_x)

# this normalization got us 64%
train_x, norms = skp.normalize(train_x, norm='max', axis=0, copy=True, return_norm=True)
test_x = test_x / norms

# this normalization got us 48%
#scaler = StandardScaler().fit(train_x)
#train_x = scaler.transform(train_x)
#test_x = scaler.transform(test_x)

# this normalization got us 57%
#min_max_scaler = skp.MinMaxScaler()
#train_x = min_max_scaler.fit_transform(train_x)
#test_x = min_max_scaler.transform(test_x)

#normalizer = skp.Normalizer().fit(train_x)
#train_x = skp.scale(train_x)
#test_x = skp.scale(test_x)



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
        name = "validation." + testing_ids[i].decode("utf-8") + ".au"
        p = predictions[i]      
        line = name + "," + genres[p] + "\n"
        f.write(line)
        i += 1
        
f.close()





