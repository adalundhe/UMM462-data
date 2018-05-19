"""
    Sean Corbett
    04/05/2018
    M462
    Homework 3: Problem 8
"""

import numpy as np
from collections import namedtuple

def sigmoid(est):
    """
        Input: X * beta (design matrix)
        Output: Estimates matrix run through sigmoid function.
    """
    return 1 / (1 + np.exp(-est))

def logLikelihood(X, Y, beta):
    """
        Input: X, Y, beta
        Output: Log-likelihood given X, Y, and beta.
    """
    est = X * beta
    likelihood = np.sum(np.dot(Y, est.T) - np.log(1 + np.exp(est)))
    return likelihood

def run_regression(X, Y, epsilon):
    """
        Runs logistic regression given X, Y, and epsilon using log-likelihood.
    """
    count = 1
    n, q = np.shape(D.X)
    beta = np.matrix(np.zeros(shape = (q, 1)))
    y_n, y_n_1 = 0, logLikelihood(X, Y, beta)

    while(abs(y_n - y_n_1)) > epsilon:
        s = np.dot(X, beta)
        pi = sigmoid(s)
        error = Y - pi

        derivative_mtx = np.dot(X.T, error)

        beta += epsilon*derivative_mtx
        y_n, y_n_1 = y_n_1, (logLikelihood(X, Y, beta) /n)
        count += 1

    return beta

def breastCancerData(path, dataSet):
    f = open(path+'breast-cancer-wisconsin.txt','r')
    data = f.read()
    f.close()
    records = data.split('\n')

    n = len(records)-1
    p = len(records[0].split(','))-2
    q = p + 1

    Y = np.matrix(np.zeros(shape = (n, 1)))
    X = np.matrix(np.ones(shape = (n, q)))
    labels = [0]*n

    counter = 1

    for i, line in enumerate(records):
        record = line.split(',')

        try:
            labels[i] = int(record[p+1]=='4')
            Y[i] = labels[i]
            X[i,1:] =  [int(x) for x  in record[1:p+1]]

        except(ValueError,IndexError):
            pass

    data = dataSet(X, Y)
    return data

def assess_accuracy(X, Y, beta):

    count, match_count = 0, 0

    p, yhat = [], []

    [yhat.append(0) if sigmoid(x * beta) < 0.5 else yhat.append(1) for x in X]

    for y in Y:
        p.append((yhat[count], y))
        count += 1


    match_count = sum([1 for y_hat, y in p if y_hat == y])
    accuracy = match_count / len(p)

    return accuracy

dataSet = namedtuple('trainingSet','X Y')

path = './data/'
D = breastCancerData(path, dataSet)

X = D.X
Y = D.Y
n, q = np.shape(D.X)

beta = run_regression(X, Y, 1e-4)
accuracy = assess_accuracy(X, Y, beta)
print("Accuracy of Estimates:", accuracy)
