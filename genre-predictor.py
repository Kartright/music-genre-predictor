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
