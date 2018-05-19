"""
Created on Sun Mar 18 10:22:03 2018
@author: brian
"""
import numpy as np
from collections import namedtuple

def initialize(g, X, fns, dfns):
    initialVars = namedtuple('variables','yHat xList hList gList zpList')
    a = .1
    xList = []
    hList = []
    gList = []
    zpList = []
    Xr = X.copy()

    m = len(g) - 1

    for r in range(m):
        xList.extend([Xr])
        if r > 0:
            shape = g[r] + 1, g[r+1]
            A = augment(Xr,0)
        else:
            shape = g[r], g[r+1]
            A = Xr

        H = np.matrix(np.random.uniform(-a, a, shape ))
        hList.extend([H])
        gList.extend([a * np.matrix(np.ones(shape))])
        AH = A * H
        zpList.extend([ dfns[r](AH) ])
        Xr = fns[r](AH)

    initialList = initialVars(Xr, xList, hList, gList, zpList)
    return initialList

def augment(X, value):
    n, _ = np.shape(X)
    return np.hstack((value*np.ones(shape= (n,1)), X ))

def dim(X):
    return np.shape(X)

def fProp(xList, hList, fns, dfns, zpList):
    m = len(xList)
    A = xList[0]

    for r in range(m):
        if r > 0:
            A = augment(xList[r], 1)
        AH = A * hList[r]

        if r < m - 1:
            xList[r+1] = fns[r](AH)
        zpList[r] = dfns[r](AH)

        yHat = fns[m-1](AH)

    return xList, zpList, yHat


def gradComputerOne(gList, xList, zpList, dEdyhat):
    k = 0

    shape = np.shape(gList[k])

    A = xList[k]

    for j in range(shape[1]):
        for i in range(shape[0]):
            dyhatdhK = np.multiply(A[:,i], zpList[k][:,j])
            gList[k][i,j] = dyhatdhK.T * dEdyhat[:,k]

    return gList

def testAcc(testData, hList, fns):
    m = len(hList)
    A = testData.X

    yHat = None

    for r in range(m):
        if r > 0:
            A = augment(zAH, 1)
        AH = A * hList[r]
        zAH = fns[r](AH)

    yHat = zAH

    return rSqr(testData.Y, testData.Y - yHat)


def getCVsample(D, sampleID, k):
    cvData = namedtuple('data','X Y')
    cvPartition = namedtuple('data', 'R E')
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if i not in sIndex]
    split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:]), cvData(D.X[sIndex,:], D.Y[sIndex,:]) )
    return split


def parkinsonsData(path):

    dataSet = namedtuple('data','X Y meanY stdY labels')
    records =  open(path,'r').read().split('\n')
    variables = records[0].split(',')

    iX = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    iY = [4, 5]

    print('Predictor variables:')
    for i in range(len(iX)) :
        print(iX[i], variables[iX[i]])
    print('Target variables:')
    for i in range(len(iY)) :
        print(iY[i], variables[iY[i]])

    n = len(records)-1
    p = len(iX) + 1
    try:
        s = len(iY)
    except(TypeError):
        s = 1

    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p )))
    for i, j in enumerate(np.arange(1,n+1,1)):
        lst = records[j].split(',')
        for k in range(s):
            Y[i,k] = float(lst[iY[k]])
        for k in range(p-1):
            X[i,k+1] = lst[iX[k]]

    s = np.std(Y, axis=0)
    m = np.mean(Y, axis = 0)
    Y = (Y - m)/s

    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)

    data = dataSet(X, Y, m, s, None)
    return data

def breastCancerData(path):
    dataSet = namedtuple('data','X Y meanY stdY labels')
    f = open(path,'r')
    data = f.read()
    f.close()
    records = data.split('\n')

    n = len(records)-1
    p = len(records[0].split(','))-1
    s = 2 # number of classes
    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p)))
    labels = [0]*n
    for i, line in enumerate(records):
        record = line.split(',')

        try:
            labels[i] = int(record[p+1]=='4')
            Y[i,labels[i]] = 1
            X[i,1:] =  [int(x)/10 for x  in record[1:p+1]]

        except(ValueError,IndexError):
            pass
    s = np.std(Y, axis=0)
    m = np.mean(Y, axis = 0)
    data = dataSet(X, Y, m, s, [np.argmax(Y[i,:]) for i in range(n)])

    return data


def BostonHousing(path):
    #https://archive.ics.uci.edu/ml/datasets/housing
    dataSet = namedtuple('data','X Y meanY stdY labels')
    p = 13 + 1
    f = open(path,'r')
    D = f.read()
    records = D.split('\n')
    n = len(records) - 1
    Y = np.matrix(np.zeros(shape = (n,1)))
    X = np.matrix(np.ones(shape = (n,p)))

    endCol = [8,15,23,26,34,43,49,57,61,68,75,82,89,96]
    startCol = [0] + [col+1 for col in endCol[0:13]]

    for i, record in enumerate(records):


        try:
            for j, pair in enumerate(zip(startCol, endCol)):

                string = record[pair[0]:pair[1]]
                try:
                    X[i,j+1] = float(string)
                    #print(i,X[i,j+1] )
                except(IndexError):
                    Y[i] = float(string)
        except(ValueError):
            pass

    s = np.std(Y, axis=0)
    m = np.mean(Y, axis = 0)
    Y = (Y - m)/s

    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)

    data =  dataSet(X, Y, m, s, None)
    return data

def KingCounty(path):
    ''' from: https://www.kaggle.com/harlfoxem/housesalesprediction '''
    ''' id,date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,
    view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,
    lat,long,sqft_living15,sqft_lot15'''
    dataSet = namedtuple('data','X Y meanY stdY labels')
    f = open(path,'r')
    D = f.read()
    records = D.split('\n')
    n = len(records)

    yCol = 2
    colNames = records[0].split(',')
    xCols = [3,4,5,6,7,8,9,10,11,12,13,14,15,19,20]
    zipcodes = [98144]

    for i in xCols:
        print(colNames[i])

    for i in range(1,len(records)-1):
        data = records[i].split(',')
        zipcode = int(data[16].strip('"'))
        indicator = -1
        notFound = True
        for j,code in enumerate(zipcodes):

            if zipcode==code:
                indicator = j
                notFound = False
        if notFound:
            zipcodes.append(zipcode)
            indicator = j+1

    n = len(records) - 2
    p = len(xCols) + len(zipcodes) + 1
    Y = np.matrix(np.zeros(shape = (n,1)))
    X = np.matrix(np.zeros(shape = (n,p)))
    for i in range(1,len(records)-1):

        data = records[i].replace('"','').split(',')
        Y[i-1] = float(data[yCol])
        zipcode = int(data[16])
        X[i-1,0] = 1.0
        for j,index in enumerate(xCols):
            X[i-1,j+1] = float(data[index])
        for j,code in enumerate(zipcodes):
            if zipcode == code:
                X[i-1,15+j+1] = 1

    s = np.std(Y)
    m = np.mean(Y)
    Y = (Y - m)/s
    X[:,1:16] = (X[:,1:16] - np.mean(X[:,1:16], axis=0)) / np.std(X[:,1:16], axis=0)

    data = dataSet(X, Y, m, s, None)
    return data

def rSqr(Y, E):
    varY = np.var(Y, axis = 0)
    varE = np.var(E, axis = 0)
    lst = np.matrix.tolist(1 - varE/varY)[0]

    return [round(float(r),4) for r in  lst]
