from python_speech_features import mfcc
import scipy.io.wavefile as wav
import numpy as np
import math
from tempfile import TemporaryFile
import os
import pickle
import random
import operator

"""
Get distance between feature vectors and find neighbours
"""
def getNeighbours(trainingSet, instance, k):
    distances = []
    for i in range(len(trainingSet)):
        dist = distance(trainingSet[i], instance, k) + distance(instance, trainingSet[i], k)
        distance.append((trainingSet[i][2], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours

"""
Identify the nearest neighbours
"""
def nearestClass(neighbours):
    classVote = {}

    for i in range(len(neighbours)):
        response = neighbours[i]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    
    sorter = sorted(classVote.items(), key = operator.itemgetter[1], reverse=True)
    return sorter[0][0]

"""
Model evaluation
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return 1.0*correct/len(testSet)

data_path = "data" # path to data set
f = open("my.dat", 'wb')

for folder in os.listdir(data_path):
    i += 1
    if i == 11:
        break
    for file in os.listdir(data_path+folder):
        (rate,sig) = wav.read(data_path+folder+"/"+file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

dataset = []
def loadDataset(filename, split, trainSet, testSet):
    with open("my.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    
    for i in range(len(dataset)):
        if random.random() < split:
            trainSet.append(dataset[i])
        else:
            testSet.append(dataset[i])

trainingSet = []
testingSet = []
loadDataset("my.dat", 0.66, trainingSet, testingSet)

predictions = []
for i in range(len(testingSet)):
    predictions.append(nearestClass(getNeighbours(trainingSet, testingSet[i], 5)))

accuracy1 = getAccuracy(testingSet, predictions)
print(accuracy1)