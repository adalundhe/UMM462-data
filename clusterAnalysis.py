# -*- coding: utf-8 -*-
"""
Spyder Editor

A start into the process of cluster analysis
This is a change on my branch.
"""
import time
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sys import argv

def calculateDay(string):
    ''' Day 0 = Dec 31, 2006 '''
    year = int(string[:4]) - 2007                   
    return time.strptime(string, "%Y-%m-%d").tm_yday + year*365
    
path = '/home/brian/M462/Data/'
fileName = 'clusterData.txt'

g = open('./'+fileName, 'r')
variables = g.readline().split(',')
print(len(variables))
print(variables[:5])

dataDict = OrderedDict()
data = g.read().split('\n')
for record in data:
    lst = record.split(',')
    x = [float(s) for s in lst[1:]]
    try:
        1/len(x)
        day = calculateDay(lst[0])
        dataDict[day] = x
    except(ZeroDivisionError):
        pass



days = list(dataDict.keys())
n = len(dataDict)
p = len(dataDict[days[0]]) 
X = np.zeros(shape = (n, p))
print(X.shape)

''' Fill the matrix with values:'''
for i, (day, values) in enumerate(dataDict.items()):
    X[i,:] = values


''' Use a small bit of data: the first 5 stocks '''
''' we're calling the function that carries out hier. agglom. clustering using complete linkage '''
''' Using the transpose of X. The transpose will be set with colums = stocks, and '''
''' rows = days '''

Z = linkage(X.T[:,:5], 'complete')
print(Z.shape)
for i in range(20):
    print('\t'.join([str(z) for z in Z[i,:]]))

''' see: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial '''
''' for descriptions of the values in Z'''

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
variables[1233], variables[1868]
X[1233,:], X[1868,:]

