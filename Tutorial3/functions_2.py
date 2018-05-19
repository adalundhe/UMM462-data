import numpy as np
from collections import namedtuple

def logistic(x):
    return softPlusP(x)

def logisticP(x):
    return logistic(x)*(1- logistic(x))

def softPlusP(x):
    return 1/(1 + np.exp(-x) )

def tanhP(x):
    return 1 - np.tanh(x)**2

def RMSProp(smoothGrad, currGrad):
    rho = 0.9
    smoothGrad = np.sqrt( rho*np.power(smoothGrad, 2)  + (1 - rho)*np.power(currGrad, 2) )
    return smoothGrad

def getSample(D, sampleID, k):

    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if sampleID[i] != k]
    cvPartition = namedtuple('data', 'R E')

    cvData = namedtuple('data','X Y labels')
    if D.labels is None:

        split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:], None ), cvData(D.X[sIndex,:], D.Y[sIndex,:], None ))
    else:
        split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:], [D.labels[i] for i in rIndex]),
                            cvData(D.X[sIndex,:], D.Y[sIndex,:],[D.labels[i] for i in sIndex]) )
    return split
