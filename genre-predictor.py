from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import math
from tempfile import TemporaryFile
import os
import pickle
import random
import operator

"""


Parameters:
    instance1   -
    instance2   -
    k           -
Returns:
    distance    -
"""
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

"""
Get distance between feature vectors and find neighbours

Paramters:
    trainingSet - data set of training inputs
    instance    - 
    k           - 
Returns:
    neighbours  - 
"""
def getNeighbours(trainingSet, instance, k):
    distances = []
    for i in range(len(trainingSet)):
        dist = distance(trainingSet[i], instance, k) + distance(instance, trainingSet[i], k)
        distances.append((trainingSet[i][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours

"""
Identify the nearest neighbours

Parameters:
    neighbours  - 
Returns:
    sorter      - 
"""
def nearestClass(neighbours):
    classVote = {}

    for i in range(len(neighbours)):
        response = neighbours[i]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

"""
Model evaluation

Parameters:
    testSet     - array of randomly selected testing data
    predictions - array of predictions for each test input
Returns:
    accuracy    - % of test values that were correctly classified by the model
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return 1.0*correct/len(testSet)

"""
Loads dataset from file and randomly splits it into training and testing datasets

Parameters:
    filename    - name of file that stores the properites of each music file
    splits      - value between 0-1 that tells how to split the data between testing and training sets
    dataset     - array of all audio files
    trainSet    - array of randomly selected training audio files
    testSet     - array of randomly selected testing audio files
Returns:
    N/A
"""
def loadDataset(filename, split, dataset, trainSet, testSet):
    with open(filename, 'rb') as f:
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



def main():
    data_path = "data/genres_original/" # path to data set
    f = open("my.dat", 'wb')    # open file to store dataset information

    i = 0
    for folder in os.listdir(data_path):
        i += 1
        if i == 11: # Only 10 genres in training set
            break
        for file in os.listdir(data_path+folder):   # Loop through each file in a genre folder
            try:
                # Get audio file properties and store information in my.dat file
                (rate,sig) = wav.read(data_path+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except ValueError:
                # Skip invalid files
                print(f"Could not read {file}, skipping...")

    f.close()   # Close my.dat file

    # Create data set partitions and load the data
    dataset = []
    trainingSet = []
    testingSet = []
    loadDataset("my.dat", 0.66, dataset, trainingSet, testingSet)

    # Get predictions for testing data using k-nearest neighbours
    predictions = []
    for i in range(len(testingSet)):
        predictions.append(nearestClass(getNeighbours(trainingSet, testingSet[i], 5)))

    # Calculate accuracy of model
    accuracy1 = getAccuracy(testingSet, predictions)
    print(accuracy1)

if __name__ == "__main__":
    main()