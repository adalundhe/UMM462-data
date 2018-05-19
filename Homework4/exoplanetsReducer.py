# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 08:35:25 2018

@author: brian
"""
import sys
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
dataSet = namedtuple('trainingSet','X Y')

def reduceData(path):
    inputData = namedtuple('data','Label Series')
    dataDict = {}
    exoPlanets = []

    with open(path, 'r') as g:
        variableNames = g.readline()
        #print(variableNames.split(',')[:10])
        for k, record in enumerate(g):
            data = record.strip('\n').split(',')
            dataDict[k] = inputData(int(data[0]), [float(x) for x in data[1:]] )
            if int(data[0]) is 2:
                exoPlanets.append(k)

    labelTable =[0]*2
    for value in dataDict.values():
        labelTable[value.Label==2] += 1
    print('N exoplanet stars = ',labelTable[1], 'N stars w/out exoplanets = ',
          labelTable[0])

    inc = .20
    pVec = [p for p in np.arange(.5*inc, 1 - .5*inc, inc)]

    N = len(dataDict)
    dataSet = namedtuple('data','X Y')

    Y = np.matrix(np.zeros( shape = (N, 1) ) )
    X = np.matrix(np.ones(shape = (N, len(pVec) + 1)))
    for key, (group, y) in dataDict.items():

        Y[key] = group - 1
        xp = np.percentile(y,[25,75])
        trY = [val for val in y if  xp[0] < val < xp[1] ]
        y = np.percentile( (y - np.mean(trY)) / np.std(trY), pVec)
        X[key,1:] = y
    D = dataSet(X, Y)
    return D

def dim(X):
    return np.shape(X)
        
