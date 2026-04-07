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

"""
Creates cutom .dat file and loads it into a testing dataset

Parameters:
    testPath    - Path to testing data directory
    testSet     - Array to store the test dataset
Returns:
    N/A
"""
def customTest(testPath, testSet):
    f = open("my-custom.dat", "wb")
    
    # Load audio files that have not been sorted into genres
    for file in sorted(os.listdir(testPath)):
        if not os.path.isfile(testPath+file):
            continue
        try:
            # Get audio file properties and store information in my.dat file
            (rate,sig) = wav.read(testPath+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, file, -1)
            pickle.dump(feature, f)
        except ValueError:
            # Skip invalid files
            print(f"Could not read {file}, skipping...")
    
    # Load files that have been sorted by genre
    i = 0
    for folder in sorted(os.listdir(testPath)):
        if os.path.isfile(testPath+folder):
            continue
        i += 1
        if i == 11: # Only 10 genres in training set
            break
        for file in os.listdir(testPath+folder):   # Loop through each file in a genre folder
            try:
                # Get audio file properties and store information in my.dat file
                (rate,sig) = wav.read(testPath+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, file, i)
                pickle.dump(feature, f)
            except ValueError:
                # Skip invalid files
                print(f"Could not read {file}, skipping...")    

    with open("my-custom.dat", 'rb') as f: # open the file containing dataset properties
        while True:
            try:
                testSet.append(pickle.load(f))  # put data file info into dataset array
            except EOFError:
                f.close()
                break

"""
Creates new dat file from default audio data

Parameters:
    N/A
Returns:
    N/A
"""
def createDefaultData():
    data_path = "data/genres_original/" # path to data set
    if (not os.path.isfile("my.dat")):  # check if my.dat file already exists
        # my.dat file does not exist, generate new file
        f = open("my.dat", 'wb')    # open file to store dataset information

        i = 0
        for folder in sorted(os.listdir(data_path)):
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

"""
Print the program menu
"""
def printMenu():
    print("=" * 40)
    print("   Music Genre Recognition Program")
    print("=" * 40)
    print("1. Default Train & Test (Random Split)")
    print("2. Default Train & Test (Genre Split)")
    print("3. Custom Test")
    print("=" * 40)
    print("Enter your choice (1-3) or enter q to quit: ", end="")


def main():
    cont = 1
    options = ['1', '2', '3', 'q']
    while(cont):
        printMenu()
        choice = input()

        while (choice not in options):
            print("INVALID INPUT, Please enter your choice (1-3) or enter q to quit: ", end="")
            choice = input()
        
        # Load default data into my.dat
        createDefaultData()
        dataset = []
        loadDataset("my.dat", dataset)
        
        if (choice == '1'): # DEFAULT TRAIN & TEST (RANDOM SPLIT)
            # Create train test split
            trainingSet = []
            testingSet = []
            randomSplit(0.66, dataset, trainingSet, testingSet)
            # Get preditions and accuracy
            predictions = []
            for i in range(len(testingSet)):
                predictions.append(nearestClass(getNeighbours(trainingSet, testingSet[i], 5)))
            accuracy = getAccuracy(testingSet, predictions)
            print(f"ACCURACY: {accuracy}")

        elif (choice == '2'): # DEFAULT TRAIN & TEST (GENRE SPLIT)
            # Create train test split
            trainingSet = []
            testingSet = []
            splitByGenre(0.66, dataset, trainingSet, testingSet)
            # Get preditions and accuracy
            predictions = []
            for i in range(len(testingSet)):
                predictions.append(nearestClass(getNeighbours(trainingSet, testingSet[i], 5)))
            accuracy = getAccuracy(testingSet, predictions)
            print(f"ACCURACY: {accuracy}")

        elif (choice == '3'): # CUSTOM TEST
            # Get testing data
            path = input("Enter path to data: ")
            testingSet = []
            customTest(path, testingSet)
            # Custom test predictions
            predictions = []
            sortedTestingSet = []
            genres = list(sorted(os.listdir("data/genres_original/")))
            print("\n=============================================")
            for i in range(len(testingSet)):
                print(testingSet[i][2],end=": ")
                genreIdx = nearestClass(getNeighbours(dataset, testingSet[i], 5))
                if (testingSet[i][-1] != -1):
                    # Only add predictions for custom files that have been sorted and add to sorted testing set
                    sortedTestingSet.append(testingSet[i])
                    predictions.append(genreIdx)
                print(genres[genreIdx-1])
            print("=============================================")
            accuracy = getAccuracy(sortedTestingSet, predictions)
            print(f"ACCURACY: {accuracy}")

        elif (choice == 'q'): # QUIT PROGRAM
            cont = 0

    return 0

if __name__ == "__main__":
    main()