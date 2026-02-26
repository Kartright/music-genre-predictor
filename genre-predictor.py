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
Get distance between two feature vectors

Parameters:
    instance1   - feature vector 1
    instance2   - feature vector 2
    k           - number of nearest neighbours to find
Returns:
    distance    - distance between the two feature vectors
"""
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1 )) 
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

"""
Get distance between feature vectors and find neighbours

Paramters:
    trainingSet - data set of training inputs
    instance    - single instance of testing set
    k           - number of nearest neighbours to get
Returns:
    neighbours  - k nearest training points to the given test instance
"""
def getNeighbours(trainingSet, instance, k):
    distances = []  # Distance to each training data point
    for i in range(len(trainingSet)):   # Get distance from test instance to each training point
        dist = distance(trainingSet[i], instance, k) + distance(instance, trainingSet[i], k)
        distances.append((trainingSet[i][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for i in range(k):  # Get the k nearest training points for the test instance
        neighbours.append(distances[i][0])
    return neighbours

"""
Identify the nearest neighbours

Parameters:
    neighbours      - Array of k nearest neighbours to a given test input
Returns:
    sorter[0][0]    - Class that appears the most in the neighbours array
"""
def nearestClass(neighbours):
    classVote = {}

    for i in range(len(neighbours)):    # Get total count of each genre that appears in the neighbours
        response = neighbours[i]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0] # Return the dominant class in the nearest neighbours list

"""
Model evaluation

Parameters:
    testSet     - array of randomly selected testing data
    predictions - array of predictions for each test input
Returns:
    accuracy    - % of test values that were correctly classified by the model
"""
def getAccuracy(testSet, predictions):
    correct = 0 # Total count of correct predictions
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return 1.0*correct/len(testSet) # Return % of test inputs correctly guessed

"""
Loads dataset from file and randomly splits it into training and testing datasets

Parameters:
    filename    - name of file that stores the properites of each music file
    dataset     - array of all audio file features

Returns:
    N/A
"""
def loadDataset(filename, dataset):
    with open(filename, 'rb') as f: # open the file containing dataset properties
        while True:
            try:
                dataset.append(pickle.load(f))  # put data file info into dataset array
            except EOFError:
                f.close()
                break

"""
Randomly splits dataset into training and testing sets based on split

Parameters:
    splits      - value between 0-1 that tells how to split the data between testing and training sets
    dataset     - array of all audio file features
    trainSet    - array of randomly selected training audio files
    testSet     - array of randomly selected testing audio files
Returns:
    N/A
"""
def randomSplit(split, dataset, trainSet, testSet):
    # Randomly split the dataset into testing and training data based on split value
    for i in range(len(dataset)):  
        if random.random() < split:
            trainSet.append(dataset[i])
        else:
            testSet.append(dataset[i])

"""
Splits dataset into training and testing sets, ensuring even spread by genre

Parameters:
    splits      - value between 0-1 that tells how to split the data between testing and training sets
    dataset     - array of all audio file features
    trainSet    - array of randomly selected training audio files
    testSet     - array of randomly selected testing audio files
Returns:
    N/A
"""
def splitByGenre(split, dataset, trainSet, testSet):
    for i in range(10):
        for k in range(len(dataset)):
            if dataset[k][2] == i:
                if random.random() < split:
                    trainSet.append(dataset[k])
                else:
                    testSet.append(dataset[k])


def main():
    data_path = "data/genres_original/" # path to data set
    if (not os.path.isfile("my.dat")):  # check if my.dat file already exists
        # my.dat file does not exist, generate new file
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
    loadDataset("my.dat", dataset)

    trainingSetRand = []
    testingSetRand = []
    randomSplit(0.66, dataset, trainingSetRand, testingSetRand)

    trainingSetEven = []
    testingSetEven = []
    splitByGenre(0.66, dataset, trainingSetEven, testingSetEven)

    # Get predictions for testing data using k-nearest neighbours
    predictionsRand = []
    for i in range(len(testingSetRand)):
        predictionsRand.append(nearestClass(getNeighbours(trainingSetRand, testingSetRand[i], 5)))

    # Get predictions for testing data using k-nearest neighbours
    predictionsEven = []
    for i in range(len(testingSetEven)):
        predictionsEven.append(nearestClass(getNeighbours(trainingSetEven, testingSetEven[i], 5)))

    # Calculate accuracy of model
    accuracy1 = getAccuracy(testingSetRand, predictionsRand)
    accuracy2 = getAccuracy(testingSetEven, predictionsEven)
    print(accuracy1)
    print(accuracy2)

if __name__ == "__main__":
    main()