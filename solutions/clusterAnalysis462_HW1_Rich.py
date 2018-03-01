# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:24:49 2018

@author: brian
"""

import time, sys
from math import factorial
from collections import namedtuple

import numpy as np
import pandas as pd
obsID = namedtuple('observation','Cluster Row')
import datetime
import matplotlib.pyplot as plt

def getData(filePath, fileName):
    
    path = filePath + fileName
    
    g = open(path, 'r')
    variables = g.readline().strip('\n').split(',')
    stocks = variables[1:]
    
    dataDict = {}
    dateDict = {}
    data = g.read().split('\n')
    for record in data:
        lst = record.split(',')
        x = [float(s) for s in lst[1:]]
        try:
            1/len(x)
            day = calculateDay(lst[0])
            dateDict[day] = lst[0]
            dataDict[day] = x
        except(ZeroDivisionError):
            pass
    
    days = list(dataDict.keys())
    nDays = len(dataDict)
    nStocks = len(dataDict[days[0]]) 
    X = np.zeros(shape = (nStocks, nDays))
    
    
    ''' Fill the matrix with values. Warning: days are NOT necessarily ordered sequentially'''
    ''' Therefore, we put them in order, and then build the matrices in sequential order:'''
    orderedDays = sorted(days)
    dates = [0]*nDays
    for i, day in enumerate(orderedDays):
        X[:, i] = dataDict[day]
        dates[i] = dateDict[day]
    
   
    ''' Scale each stock series to have mean 0 and standard deviation 1 '''
    ''' Using numpy, the operation: matrix / y divides each column of M by the corresponding element in y '''

    print('X shape = ', np.shape(X))    
    
    xTranspose = X.T
    Z = ( (xTranspose - np.mean(xTranspose, axis=0)) / np.std(xTranspose, axis=0) ).T    
    columnDict = {stock:i for i, stock in enumerate(stocks)}    
    
    return stocks, Z, columnDict, orderedDays, dates 
    
def plot(dataToPlot, dates):
    ''' Code for plotting  a time series '''
    ''' Set up time variable for x-axis '''
    x = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in dates]
    plt.figure(figsize=(10, 5))
    legend = []
    for i in range(len(dataToPlot)):
        y = dataToPlot[i]    
        plt.plot(x,y) 
        legend.append('Cluster '+str(i))
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.legend(loc='upper right')
    plt.legend(legend)
    #plt.savefig('/home/brian/M467/Figures/centroids.eps', format='eps',dpi = 1200)
    plt.show()

def calculateDay(string):
    ''' Day 0 = Dec 31, 2006 '''
    year = int(string[:4]) - 2007                   
    return time.strptime(string, "%Y-%m-%d").tm_yday + year*365

def createInitialAssigments(Z, nClusters, stocks):

    n, nDays = np.shape(Z)
    ''' compute the mean of the last 100 days for each stock '''
    means = np.mean(Z[:,nDays- 100:],axis = 1)
    orderVector = np.argsort(means)
    '''
    print('Sorted stocks according to last 100 day mean:')
    
    for i in range(n):
        print(stocks[orderVector[n-i-1]], '\t',means[orderVector[n-i-1]])
    '''
    ''' Initial assignments based on splitting the range of 100-day mean into intervals: '''
    splits = [(i+1)*int(n/nClusters) for i in range(nClusters-1)]
    splits.append(n)
    
    ''' Observations assigned to cluster: '''
    initialAssignments = [0] * n
    for i in range(n):
        initialAssignments[orderVector[i]]  = sum([j*(splits[j-1] < i <= splits[j]) for j in range(1,nClusters)] )

    ''' For reference: save the cluster assignment and row location (i) for each stock'''
    membershipDict = {stock : obsID(cluster, i) for i, (stock, cluster) in enumerate(zip(stocks, initialAssignments)) } 
    return membershipDict

def centroidComputer(membershipDict, Z):
    ''' Computes the centroid for each cluster using the data in Z '''
    ''' Iterate over stocks and aggregate the series, then divide to get the means '''
    
    '''' Initialize empty dictionary :'''
    centroidDict = dict.fromkeys(range(nClusters))  
    
    ''' identifier identifies the current cluster membership of a stock '''
    ''' and the row of Z containing the series for the stock '''
    for stock, identifier in membershipDict.items():
        center = centroidDict[identifier.Cluster]
        
        try:
            center[0] += 1
            center[1] = [a+b for a, b in zip(center[1],Z[identifier.Row,:])]
        except(TypeError): 
            center =[1, Z[identifier.Row,:]]
        centroidDict[identifier.Cluster] = center
    
    ''' Divide values by the number of members in the respective cluster: '''  
    ''' centroidDict[i][0] is the number of members for cluster i '''
    ''' centroidDict[i][1] is the series of sums for cluster i '''
    
    for i in range(nClusters):
        centroidDict[i][1] = [x/centroidDict[i][0] for x in centroidDict[i][1]]
    return centroidDict


def kMeansClusters(Z, membershipDict, centroidDict):

    ''' Total SS is the error sums-of-squares with no clusters '''
    xbar = np.mean(Z, axis = 0)
    totalSS = sum([sum((z-xbar)**2) for z in Z])
    print('\nObj fn = ',round(totalSS,1))
    
    ''' Iterate until no observations (stocks) are relocated '''
    ''' Dumb algorithm --- rebuild the clusters according to the nearest centroid to '''
    ''' each observation '''
    
    repeat = True
    while repeat == True:
        updateDict = {}
        ''' Iterate over stocks '''
        for stock, ID  in membershipDict.items():
            minDist = 1E5 # Initialize the minimum obs-to-cluster distance.
            
            ''' Iterate over cluster and determine the nearest centroid/cluster'''
            for label, center in centroidDict.items():
                centroid = center[1]
                
                ''' cos distance '''
#                zToCenDist = 0.5*(1 - np.corrcoef(Z[ID.Row,:], centroid)[0,1])
                ''' Euclidean distance squared '''
                zToCenDist = sum((Z[ID.Row,:] - centroid)**2)
    
                ''' Reassign membership if necessary:'''
                if zToCenDist < minDist: 
                    nearestCluster = label
                    minDist = zToCenDist          
                 
            ''' Save the results for the stock'''       
            updateDict[stock] = obsID(nearestCluster, ID.Row)     
    
        ''' Test whether there's been an update. '''
        ''' If so, recompute the centroids and the objective function '''
        
        if membershipDict != updateDict:
            ''' MUST use copy. Otherwise membershipDict and updateDict share the same address '''
            membershipDict = updateDict.copy()
            ''' Recompute the centroids:'''
            centroidDict = centroidComputer( membershipDict, Z)
            
            objFn = 0
            for cluster, row in membershipDict.values():
                ''' euclidean dist squared '''
                objFn += sum((centroidDict[cluster][1] - Z[row,:])**2)
                ''' cos distance '''
#                objFn += 0.5*(1 - np.corrcoef(Z[ID.Row,:], centroidDict[cluster][1])[0,1])
             
            print('UPDATE Obj fn = ', round(objFn,1))
        else:
            repeat = False    
            print('No UPDATE')
            
        for label, values in centroidDict.items():
            print('Cluster =', label,' Size = ',values[0])
    return membershipDict, centroidDict

'''*************************** MY FUNCTIONS ***************************'''

'''############################## PART 1 ##############################'''

# build a multi-dim array that lists column indices of each cluster
def buildClusterRowList(membershipDict, centroidDict):
    # make a list for each cluster in clusterColIndex
    clusterRowList = []
    clusterRowList = [[] for cluster in centroidDict.keys()]
    # add each stock's column-index to the clusterColIndex according to it's cluster
    for stock in membershipDict:
        cluster = membershipDict[stock][0]
        row = membershipDict[stock][1]
        clusterRowList[cluster].append(row)
    
    return clusterRowList

# (f1) average Euclidean Distance between all stock pairs (i,j) over nDays of observations
def f1ClusterVarCalc(Z, clusterRowList, centroidDict, nDays):
    ''' make dict to hold variance of each cluster '''
    clusterVarDict = {}
    totalDays = Z.shape[1]
    
    ''' iterate over clusters '''
    for clusterIdx in range(len(clusterRowList)):
        clusterVarDict[clusterIdx] = 0
        currentCluster = clusterRowList[clusterIdx]
        nStocks = len(currentCluster)
        
        ''' iterate over pairs of stocks (i,j) '''
        for i in range(nStocks):
            ''' number of (i,j) pairs in cluster size n = C(n,2) '''
            ij_combinations = factorial(nStocks)/(2*factorial(nStocks-2))
            
            for j in range(i+1, nStocks):
                ij_dist = 0
                
                ''' look up row index '''
                i_row = currentCluster[i]
                j_row = currentCluster[j]
                
                ''' iterate over days '''
                for day in range(totalDays-nDays, totalDays):
                    i_stockVal = Z[i_row][day]
                    j_stockVal = Z[j_row][day]
                    ''' use square diff for each day '''
                    ij_dist += (i_stockVal - j_stockVal)**2
                ''' take sqrt of square distances (Euclidean Dist) '''
                ij_dist = np.sqrt(ij_dist)
                ''' add it to cluster total '''
                clusterVarDict[clusterIdx] += ij_dist
                
        ''' divide total cluster distance by number of (i,j) combinations in cluster '''
        clusterVarDict[clusterIdx] /= ij_combinations
        
    totalVar = 0
    
    for cluster in clusterVarDict:
        totalVar += clusterVarDict[cluster]
                
    return clusterVarDict, totalVar

# (f2) average Euclidean Distance between all stock pairs (i,j) over nDays of observations
def f2ClusterVarCalc(Z, clusterRowList, centroidDict, nDays):
    ''' make dict to hold variance of each cluster '''
    clusterVarDict = {}
    totalDays = Z.shape[1]
    
    ''' iterate over clusters '''
    for clusterIdx in range(len(clusterRowList)):
        clusterVarDict[clusterIdx] = 0
        currentCluster = clusterRowList[clusterIdx]
        nStocks = len(currentCluster)
        
        ''' iterate over pairs of stocks (i,j) '''
        for i in range(nStocks):
            ''' number of (i,j) pairs in cluster size n = C(n,2) '''
            ij_combinations = factorial(nStocks)/(2*factorial(nStocks-2))
            
            for j in range(i+1, nStocks):
                ij_cosine = 0
                
                ''' look up row index '''
                i_row = currentCluster[i]
                j_row = currentCluster[j]
                
                ''' use cosine distance for each pair '''
                i_stockVal = Z[i_row][(totalDays-nDays):totalDays]
                j_stockVal = Z[j_row][(totalDays-nDays):totalDays]
                
                ij_cosine = abs(np.corrcoef(i_stockVal, j_stockVal))[0][1]
                ij_cosine += (1-ij_cosine)/2
                    
                ''' add it to cluster total '''
                clusterVarDict[clusterIdx] += ij_cosine
                
        ''' divide total cluster distance by number of (i,j) combinations in cluster '''
        clusterVarDict[clusterIdx] /= ij_combinations
        
    totalVar = 0
    
    for cluster in clusterVarDict:
        totalVar += clusterVarDict[cluster]
                
    return clusterVarDict, totalVar

# (f3) average Euclidean Distance between all each stock and the centroid
def f3ClusterVarCalc(Z, clusterRowList, centroidDict, nDays):
    ''' make dict to hold variance of each cluster '''
    clusterVarDict = {}
    totalDays = Z.shape[1]
    
    ''' iterate over clusters '''
    for clusterIdx in range(len(clusterRowList)):
        clusterVarDict[clusterIdx] = 0
        currentCluster = clusterRowList[clusterIdx]
        nStocks = len(currentCluster)
        
        ''' iterate over all stocks '''
        for i in range(nStocks):
            i_dist = 0
            ''' look up row index '''
            i_row = currentCluster[i]
                
            ''' iterate over days '''
            for day in range(totalDays-nDays, totalDays):
                i_stockVal = Z[i_row][day]
                ''' calc distance from centroid '''
                i_dist += abs(i_stockVal - centroidDict[clusterIdx][1][day])
            
            ''' divide total dist between stock and centroid by number of days '''
            i_dist /= nDays
            
            clusterVarDict[clusterIdx] += i_dist
            
        ''' divide total cluster dist by number of stocks '''
        clusterVarDict[clusterIdx] /= n
        
    totalVar = 0
    
    for cluster in clusterVarDict:
        totalVar += clusterVarDict[cluster]
                
    return clusterVarDict, totalVar

'''############################## PART 3 ##############################'''

# build a binary data table - 1 if x_i+1 > x_i, 0 otherwise
def buildBinaryZ(Z):
    binZ = np.zeros((Z.shape[0], Z.shape[1]-1))
    
    for stock in range(Z.shape[0]):
        for day in range(Z.shape[1]-1):
            if Z[stock][day+1] > Z[stock][day]:
                binZ[stock][day] = 1
                
    return binZ

''' #################################################################### '''
        
'''************************   Program starts here ************************'''         

nClusters = 4

filePath = '/Users/Devreckas/Dropbox/College-Courses/2018-EARLY-SPRING/Theoretical-Data-Analytics/data/'
fileName = 'clusterDataSM.txt'
fileName = 'clusterDataLG.txt'

pathName = filePath + fileName
stocks, Z, columnDict, days, dates  = getData(filePath, fileName)

'''############################## PART 3 ##############################'''

''' Compute Hamming Dist matrix '''
#binZ = buildBinaryZ(Z)
#Z = binZ

'''####################################################################'''

n, nDays = np.shape(Z)
print('N days = ', nDays, '\nN stocks = ', n,'\nZ shape = ',n,'X', nDays,'\n')
nStocks = len(stocks)

''' Initialize ... '''
membershipDict = createInitialAssigments(Z, nClusters, stocks)

''' Compute centroids and plot a few '''
centroidDict = centroidComputer( membershipDict, Z)
dataToPlot= [centroidDict[0][1],centroidDict[1][1],centroidDict[2][1]]
plot(dataToPlot, dates)

'''  Call the k-means function '''
membershipDict, centroidDict = kMeansClusters(Z, membershipDict, centroidDict)

'''############################## PART 1 ##############################'''

clusterRowList = buildClusterRowList(membershipDict, centroidDict)

# Calculate all variability statistics
###
# Calc var using euclidean distance
f1ClusterVarDict, f1Total = f1ClusterVarCalc(Z, clusterRowList, centroidDict, 10)
print('F1:',f1ClusterVarDict)
# Calc var using cosine distance
f2ClusterVarDict, f2Total = f2ClusterVarCalc(Z, clusterRowList, centroidDict, 10)
print('F2:', f2ClusterVarDict)
# Calc var using distance from centroid
f3ClusterVarDict, f3Total = f3ClusterVarCalc(Z, clusterRowList, centroidDict, 10)
print('F3:', f3ClusterVarDict)

# Build table of variability of clusters
varTbl = pd.DataFrame(columns=['Cluster','Num of Members','Var F1', 'Var F2', 'Var F3'])

row = 0
for cluster in centroidDict:
    varTbl.loc[row] = [cluster, centroidDict[cluster][0], f1ClusterVarDict[cluster], f2ClusterVarDict[cluster], f3ClusterVarDict[cluster]]
    row += 1

print(varTbl)

'''############################## PART 2 ##############################'''

# Build table of centroid distances by day
cols = []
cols = [i for i in range(len(centroidDict.keys()))]
cols.append('Day')
dist2dayTbl = pd.DataFrame(columns=cols)

for day in range(nDays):
    entry = [centroidDict[cluster][1][day] for cluster in centroidDict.keys()]
    entry.append(day)
    dist2dayTbl.loc[day] = entry
    
print(dist2dayTbl)
dist2dayTbl.plot(x='Day', y=[i for i in range(len(centroidDict.keys()))])
sys.exit(0)

'''############################## PART 3 ##############################'''



'''####################################################################'''

''' Plot the new centroids '''
dataToPlot= [centroidDict[0][1],centroidDict[1][1],centroidDict[2][1]  ]
plot(dataToPlot, dates)

''' Example of exponential smoothing '''
''' Be careful that the data to be smoothed are in chronological order '''
y = centroidDict[0][1]

len(y)
a = .25
wts = [a*(1 - a)**i for i in range(nDays)]
smooth = [0]*nDays

for i, iday in enumerate(dates):

    distances = [abs(calculateDay(iday) - calculateDay(date)) for date in dates]

    orderVector = np.argsort(distances)

    smooth[i]  = sum([wts[k]*y[j] for k, j in enumerate(orderVector)])

    #print(iday, distances[:7], orderVector[:7], smooth[i] )

    if i  ==100: break

dataToPlot = [y[:100], smooth[:100]]

plot(dataToPlot, dates[:100])

print('...completed')