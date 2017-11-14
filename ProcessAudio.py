import os
import numpy as np
import scipy
import soundfile as sf
import speechpy as sp
import scipy.stats as stats
from scipy import signal


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
        print("finished processing "+genre+"...")
    print("Finished processing training data")
    dir = os.path.join(os.getcwd(), 'data\\validation\\')
    for file in os.listdir(dir):
        if file.endswith(".au"):
            data, samplerate = sf.read(dir + "\\" + file)
            fileid = file.split(".")[1]
            ExtractTestingFeatures(data, samplerate, fileid)
    print("Finished processing testing data")
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
    fourier = scipy.fft(data)
    #extract first 1000 fft featues
    fft_features = np.abs(fourier[:1000])
    #extract fft feature
    mfcc = cepsFeatures(data,samplerate)
    #extract entropy features
    # extract bpm features
    bpm = bpmFeatures(data)
    features = np.concatenate((fft_features, mfcc, bpm), axis=0)
    train_x.append(features)
    #save the genre for later
    train_y.append(genre)

def ExtractTestingFeatures(data,samplerate,id):
    fourier = scipy.fft(data)
    #extract the fft feature
    fft_features = np.abs(fourier[:1000])
    # extract mfcc feature
    mfcc = cepsFeatures(data,samplerate)
    #extract entropy Features
    #extract bpm features
    bpm = bpmFeatures(data)
    features = np.concatenate((fft_features, mfcc, bpm), axis=0)
    #append the features to a 2d array
    test_x.append(features)
    #append the id to an array for later
    test_y.append(id)

def cepsFeatures(data, samplerate):
    #returns the mfcc features of the data
    ceps = sp.mfcc(data, samplerate)
    numberOfCeps = len(ceps)
    return np.mean(ceps[int(numberOfCeps * 1 / 10):int(numberOfCeps * 9 / 10)], axis=0)

def filterBankFeatures(data,samplerate):
    cube = sp.lmfe(data, sampling_frequency=samplerate, frame_length=0.020, frame_stride=0.01,
                              num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    #cube = sp.extract_derivative_feature(logenergy)
    features = np.mean(cube[int(len(cube) * 1 / 10):int(len(cube) * 9 / 10)], axis=0)
    return features

def bpmFeatures(data):
    #Our third and custom feature attempts to extract tempo
    #This is a naive attempt at extracting tempo data from the song
    data = np.abs(data)
    #get the mean
    high_en = np.mean(data)
    peaks = []
    for i in range(0, len(data)):
        if data[i] > high_en:
            #take all the points above the mean
            peaks.append(i)
    differences = []
    for i in range(0, len(peaks)-1):
        #take the distance between peaks
        diff = np.abs(peaks[i] - peaks[i+1])
        differences.append(diff)
    features = []
    #take the mean of the distance between peaks
    features.append(np.mean(differences))
    return features

def entropyFeatures(fourier):
    #chunk into 50 different arrays of appx 13,236 each
    i = 0
    arrayindex = 0
    chunks = [] * 1000
    for i in range(0, len(fourier), 6168):
        list = fourier[i:i+6168]
        chunks.append(list)
    entropies = [] * 1000
    for i in range(0, len(chunks)):
        p = (1 / len(chunks[i])) * (np.power(np.abs(chunks[i]), 2))
        p = p / np.sum(p)
        entropy = stats.entropy(p)
        entropies.append(entropy)

    features = [] * 4
    sd_entropies = np.std(entropies)
    mean_entropies = np.mean(entropies)
    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)
    features.append(sd_entropies)
    features.append(mean_entropies)
    features.append(min_entropy)
    features.append(max_entropy)
    return features

print("Processing data...")
ProcessData()
