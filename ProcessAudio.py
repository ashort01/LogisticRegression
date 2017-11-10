import os
import numpy as np
import scipy
import soundfile as sf
import speechpy as sp
from sklearn import linear_model
from sklearn import metrics

genres = {"blues":0,"classical":1,"country":2,
          "disco":3, "hiphop":4, "jazz":5,
          "metal":6,"pop":7, "reggae":8,
          "rock":9}	  

train_x = []
train_y = []
test_x = []
test_y = []
		  
def ProcessData():
    for genre in genres.keys():
        dir = os.path.join(os.getcwd(), 'data\\genres\\'+genre)
        fileIndex = 0
        for file in os.listdir(dir):
            if file.endswith(".au"):
                data, samplerate = sf.read(dir+"\\"+file)
                ExtractFeatures(data,samplerate,genres[genre],fileIndex)
                fileIndex += 1


def ExtractFeatures(data,samplerate,genre,index):
    #extract first 1000 fft featues
    fft_features = abs(scipy.fft(data)[:1000])
    train_x.append(fft_features)
    train_y.append(genre)
    if index > 80:
        test_x.append(fft_features)
        test_y.append(genre)
	
    #extract MFCC features
    ceps =sp.mfcc(data,samplerate)
    numberOfCeps = len(ceps)
    x = np.mean(ceps[int(numberOfCeps*1/10):int(numberOfCeps*9/10)],axis=0)

    #run data through filterbank
    ofb = pf.FractionalOctaveFilterbank(samplerate,4)
    y, states = ofb.filter(data)
    L = (10 * np.log10(np.sum(y * y, axis=0)))[:30]


print "Processing data..."
ProcessData()
print "Creating model..."
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(train_x, train_y)
print "Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x))
print "Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x))