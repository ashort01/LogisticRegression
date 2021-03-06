import pandas as pd
import csv
import itertools
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

train_x = pd.read_csv('examples.csv', sep=',', header=None)
train_y = pd.read_csv('classes.csv', sep=',', header=None)
train_y = np.asarray(train_y).flatten()

test_x = pd.read_csv('testing.csv', sep=',', header=None)
# test_y = pd.read_csv('test_y.csv', sep=',',header=None)
# test_y = np.asarray(test_y).flatten()


# this normalization got us 40%
#scaler= StandardScaler()
#train_x = scaler.fit_transform(train_x)
#test_x = scaler.transform(test_x)

# this normalization got us 42%
train_x, norms = skp.normalize(train_x, norm='max', axis=0, copy=True, return_norm=True)
test_x = test_x / norms

# this normalization got us 42%
# scaler = StandardScaler().fit(train_x)
# train_x = scaler.transform(train_x)
# test_x = scaler.transform(test_x)

# this normalization got us 40%
#min_max_scaler = skp.MinMaxScaler()
#train_x = min_max_scaler.fit_transform(train_x)
#test_x = min_max_scaler.transform(test_x)

#this submisstion go us 36%
#normalizer = skp.Normalizer().fit(train_x)
#train_x = skp.scale(train_x)
#test_x = skp.scale(test_x)



print("Creating model...")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh = neigh.fit(train_x, train_y)
# print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

predictions = neigh.predict(test_x)
testing_ids = np.genfromtxt('testing_ids.csv', dtype=object, delimiter=',')

# do output for kaggle on testing data
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


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Found on scikit-learn.org
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# predict across 10 fold validations
y_pred = cross_val_predict(neigh, train_x, train_y, cv=10)

# get the confusion matrix
conf_mat = confusion_matrix(train_y, y_pred)

# Plot the confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, genres)
plt.show()