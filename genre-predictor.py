from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import math
from tempfile import TemporaryFile
import os
import pickle
import random
import operator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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
        if testSet[i][2] == predictions[i]:
            correct += 1
    return 1.0*correct/len(testSet) # Return % of test inputs correctly guessed

"""
Plots a confusion matrix

Parameters:
    testLables  - Array of actual lables for each test value
    predictions - Array of predicted label for each test value
    labels      - Array of all labels
"""
def plotCm(testLabels, predictions, labels):
    cm = confusion_matrix(testLabels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("MLP — Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.show()

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
def genreSplit(split, dataset, trainSet, testSet):
    for i in range(1, 11):
        for k in range(len(dataset)):
            if dataset[k][2] == i:
                if random.random() < split:
                    trainSet.append(dataset[k])
                else:
                    testSet.append(dataset[k])

"""
Use one data set to train and one dataset to test
"""
def datasetSplit():
    # Load the training dataset
    datasetPath = input("Enter the training dataset name: ") + ".dat"
    while (not os.path.isfile(datasetPath)):
        datasetPath = input("INVALID NAME, Enter the training dataset name: ") + ".dat"
    trainDataset = []
    loadDataset(datasetPath, trainDataset)

    # Load the testing dataset
    datasetPath = input("Enter the testing dataset name: ") + ".dat"
    while (not os.path.isfile(datasetPath)):
        datasetPath = input("INVALID NAME, Enter the testing dataset name: ") + ".dat"
    testDataset = []
    loadDataset(datasetPath, testDataset)

     # Get K value
    while True:
        try:
            k = int(input("Enter a K value for KNN: "))
            break
        except ValueError:
            print("INVALID INPUT, please enter an int for the K value: ", end="")

    # Custom test predictions
    predictions = []
    sortedTestingSet = []
    testLabels = []
    genres = list(sorted(os.listdir("data/genres_original/")))
    print("\n=============================================")
    for i in range(len(testDataset)):
        print(testDataset[i][3],end=": ")
        genreIdx = nearestClass(getNeighbours(trainDataset, testDataset[i], k))
        if (testDataset[i][2] != -1):
            # Only add predictions for custom files that have been sorted and add to sorted testing set
            sortedTestingSet.append(testDataset[i])
            testLabels.append(testDataset[i][2])
            predictions.append(genreIdx)
        print(genres[genreIdx-1])
    print("=============================================")
    plotCm(testLabels, predictions, genres)
    accuracy = getAccuracy(sortedTestingSet, predictions)
    print(f"ACCURACY: {accuracy}")


"""
Creates new dat file from default audio data

Parameters:
    dataPath    - Path to audio data
    newFileName - Name for new .dat file that will be created
Returns:
    N/A
"""
def generateDat(dataPath, newFileName):
    newFileName = newFileName + ".dat"
    if (not os.path.isfile(newFileName)):  # check if the .dat file with newFileName already exists
        # newFileName.dat file does not exist, generate new file
        f = open(newFileName, 'wb')    # open file to store dataset information

        i = 0
        for folder in sorted(os.listdir(dataPath)):
            i += 1
            if i == 11: # Only 10 genres in training set
                break
            for file in sorted(os.listdir(dataPath+folder)):   # Loop through each file in a genre folder
                try:
                    # Get audio file properties and store information in my.dat file
                    (rate,sig) = wav.read(dataPath+folder+"/"+file)
                    if (len(sig) == 0):
                        print(f"No audio in {file}, skipping...")
                        continue
                    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                    covariance = np.cov(np.matrix.transpose(mfcc_feat))
                    mean_matrix = mfcc_feat.mean(0)
                    feature = (mean_matrix, covariance, i, file)
                    pickle.dump(feature, f)
                except ValueError:
                    # Skip invalid files
                    print(f"Could not read {file}, skipping...")

        f.close()   # Close my.dat file
    else:
        print("A .dat file with that name already exists")

"""
Print the program menu
"""
def printMenu():
    print("=" * 40)
    print("   Music Genre Recognition Program")
    print("=" * 40)
    print("1. Generate Dataset")
    print("2. Train & Test")
    print("=" * 40)
    print("Enter your choice (1-2) or enter q to quit: ", end="")


def main():
    cont = 1
    options = ['1', '2', '3', '4', '5', 'q']
    while(cont):
        printMenu()
        choice = input()

        while (choice not in options):
            print("INVALID INPUT, Please enter your choice (1-5) or enter q to quit: ", end="")
            choice = input()
        
        if (choice == '1'): # GENERATE DATASET
            path = input("Enter path to data: ")
            newFileName = input("Enter name for dataset: ")
            generateDat(path, newFileName)
            print("Dataset generated!")

        elif (choice == '2'): # TRAIN & TEST
            # Get split type
            splitType = input("1 = Random\n2 = By Genre\n3 = By Dataset\nChoose a split type: ")
            while (splitType not in ['1','2','3']):
                splitType = input("INVALID SELECTION\n1 = Random\n2 = By Genre\nChoose a split type: ")

            if (splitType == '3'):
                datasetSplit()
                continue
            
            # Load the dataset to use
            datasetPath = input("Enter the dataset name: ") + ".dat"
            while (not os.path.isfile(datasetPath)):
                datasetPath = input("INVALID NAME, Enter the dataset name: ") + ".dat"
            dataset = []
            loadDataset(datasetPath, dataset)

            # Get a split
            while True:
                try:
                    split = float(input("Enter a data split (0-1): "))
                    if 0 < split < 1:
                        break
                    print("Must be value between 0-1, try again: ", end="")
                except ValueError:
                    print("INVALID INPUT, please enter a number: ", end="")

            # Get K value
            while True:
                try:
                    k = int(input("Enter a K value for KNN: "))
                    break
                except ValueError:
                    print("INVALID INPUT, please enter an int for the K value: ", end="")

            # Create train test split
            trainSet = []
            testSet = []
            if (splitType == '1'):
                randomSplit(split, dataset, trainSet, testSet)
            elif (splitType == '2'):
                genreSplit(split, dataset, trainSet, testSet)

            # Get preditions and accuracy
            predictions = []
            testLabels = []
            for i in range(len(testSet)):
                predictions.append(nearestClass(getNeighbours(trainSet, testSet[i], k)))
                testLabels.append(testSet[i][2])
            
            genres = list(sorted(os.listdir("data/genres_original/")))
            plotCm(testLabels=testLabels, predictions=predictions, labels=genres)
            accuracy = getAccuracy(testSet, predictions)
            print(f"ACCURACY: {accuracy}")

        elif (choice == 'q'): # QUIT PROGRAM
            cont = 0

    return 0

if __name__ == "__main__":
    main()