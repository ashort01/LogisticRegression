import os
import numpy as np
import scipy
import soundfile as sf
import speechpy as sp
import scipy.stats as stats

#Dictionary of genres
genres = {"blues":0,"classical":1,"country":2,
          "disco":3, "hiphop":4, "jazz":5,
          "metal":6,"pop":7, "reggae":8,
          "rock":9}	  

train_x = []
train_y = []
test_x = []
test_y = []
		  
def ProcessData():
    #loop through the genres
    for genre in genres.keys():
        #enter the specific genre folder
        dir = os.path.join(os.getcwd(), 'data\\genres\\'+genre)
        fileIndex = 0
        #loop through all of the files in the folder
        for file in os.listdir(dir):
            if file.endswith(".au"):
                #read audio file
                data, samplerate = sf.read(dir+"\\"+file)
                #extract features from audio file, populates arrays
                ExtractFeatures(data,samplerate,genres[genre],fileIndex)
                fileIndex += 1
        print("finished processing "+genre+"...")
    print("Finished processing training data")
    #testing data directory
    dir = os.path.join(os.getcwd(), 'data\\validation\\')
    for file in os.listdir(dir):
        if file.endswith(".au"):
            #reads the file
            data, samplerate = sf.read(dir + "\\" + file)
            fileid = file.split(".")[1]
            #extracts testing data from the files
            ExtractTestingFeatures(data, samplerate, fileid)
    print("Finished processing testing data")
    #write the features to csv files
    np.savetxt("examples.csv", train_x, delimiter=",")
    with open("classes.csv", 'w') as c:
        i=0
        for x in train_y:
            i += 1
            c.write(str(x))
            if i < len(train_y):
                c.write(",")
    c.close()

    #write the validation features to csv files
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
    '''
    While this feature showed us some promise, it ended up not being used.
    It performed better on kfold testing, but not on kaggle testing.
    :param fourier:
    :return:
    '''
    #chunk into 50 different arrays of appx 13,236 each
    i = 0
    arrayindex = 0
    chunks = [] * 2000
    for i in range(0, len(fourier), 3084):
        list = fourier[i:i+6168]
        chunks.append(list)
    entropies = [] * 1000
    for i in range(0, len(chunks)):
        p = (1 / len(chunks[i])) * (np.power((np.abs(chunks[i])), 2))
        p = p / np.sum(p)
        entropy = stats.entropy(p)
        entropies.append(entropy)

    features = [] * 4
    sd_entropies = np.std(entropies)
    mean_entropies = np.mean(entropies)
    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)
    max_difference_subsequent = np.max([x - entropies[i - 1] for i, x in enumerate(entropies)][1:])
    features.append(sd_entropies)
    #features.append(mean_entropies)
    #features.append(min_entropy)
    #features.append(max_entropy)
    features.append(max_difference_subsequent)
    return features


print("Processing data...")
ProcessData()
