import pandas as pd
import itertools
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]	 

#read in data from the csv's
train_x = pd.read_csv('examples.csv', sep=',',header=None)
train_y = pd.read_csv('classes.csv', sep=',',header=None)
train_y = np.asarray(train_y).flatten()

test_x = pd.read_csv('testing.csv', sep=',',header=None)

#normalize
# this normalization was the best
train_x, norms = skp.normalize(train_x, norm='max', axis=0, copy=True, return_norm=True)
test_x = test_x / norms

print("Creating model...")
#create the lr model
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)

#predict testing data for kaggle
predictions = mul_lr.predict(test_x)
testing_ids = np.genfromtxt('testing_ids.csv', dtype=object, delimiter=',')


#do output csv for kaggle on testing data
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
                          title='Logistic Regression',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #Found on scikit-learn.org
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

#Create a shuffled kfold
cv = KFold(n_splits=10, shuffle=True)

#predict across 10 fold validations
y_pred = cross_val_predict(mul_lr,train_x,train_y,cv=cv)

#Get the accuracy scores for the cross val prediction
scores = cross_val_score(mul_lr, train_x, train_y, cv=cv)
#get the mean for overall accuracy
accuracy = scores.mean()
print("Overall Accuracy:" +str(accuracy))


#get the confusion matrix
conf_mat = confusion_matrix(train_y,y_pred)


#Plot the confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat,genres)
plt.show()