import os
import numpy as np
import scipy
import soundfile as sf
import speechpy as sp


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
    dir = os.path.join(os.getcwd(), 'data\\validation\\')
    for file in os.listdir(dir):
        if file.endswith(".au"):
            data, samplerate = sf.read(dir + "\\" + file)
            fileid = file.split(".")[1]
            ExtractTestingFeatures(data, samplerate, fileid)
    np.savetxt("examples.csv", train_x, delimiter=",")
    with open("classes.csv", 'w') as c:
        i=0
        for x in train_y:
            i += 1
            c.write(str(x))
            if i < len(train_y):
                c.write(",")
    c.close()


    np.savetxt("testing.csv",test_x, delimiter=",")
    with open("testing_ids.csv", 'w') as f:
        ids = ",".join(test_y)
        f.write(ids)
        f.close()



def ExtractFeatures(data,samplerate,genre,index):
    #extract first 1000 fft featues
    fft_features = fftFeatures(data)

    #extract fft feature
    mfcc = cepsFeatures(data,samplerate)

    #TODO: get thrid feature going
    #run data through filterbank
    #ofb = sp.FractionalOctaveFilterbank(ssamplerate,4)
    #y, states = ofb.filter(data)
    #L = (10 * np.log10(np.sum(y * y, axis=0)))[:30]


    features = np.concatenate((fft_features, mfcc), axis=0)
    train_x.append(features)
    #save the genre for later
    train_y.append(genre)

def ExtractTestingFeatures(data,samplerate,id):
    #extract the fft feature
    fft_features = fftFeatures(data)
    # extract mfcc feature
    mfcc = cepsFeatures(data,samplerate)
    features = np.concatenate((fft_features, mfcc), axis=0)
    #append the features to a 2d array
    test_x.append(features)
    #append the id to an array for later
    test_y.append(id)

def fftFeatures(data):
    #returns the fft features of the data
    return np.abs(scipy.fft(data)[:1000])

def cepsFeatures(data, samplerate):
    #returns the mfcc features of the data
    ceps = sp.mfcc(data, samplerate)
    numberOfCeps = len(ceps)
    return np.mean(ceps[int(numberOfCeps * 1 / 10):int(numberOfCeps * 9 / 10)], axis=0)

print("Processing data...")
ProcessData()
