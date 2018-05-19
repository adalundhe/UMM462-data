"""
    Sean Corbett
    04/18/2018
    M462
    Tutorial 1: Corrected
"""

from functions import *
from random import choice, seed
from neural_network import ActivationFunction
from activation_functions import (
    rlu,
    rluPrime,
    identity,
    unit
)

path = './data/parkinsons_updrs.csv'
D = parkinsonsData(path)
n, p = dim(D.X)
n, s = dim(D.Y)
print(n, p, s)

g = [p, p, s]
K = 10

z0 = ActivationFunction(rlu, rluPrime)
z1 = ActivationFunction(identity, unit)
fns = [z0.evaluate, z1.evaluate]
dfns = [z0.differentiate, z1.differentiate]

seed(0)
np.random.seed(0)

sampleID = [choice(range(K)) for i in range(n)]

for k in range(K):
    gamma = .00001

    try:
        del progress
    except(NameError):
        pass

    F = getCVsample(D, sampleID, k)
    X = F.R.X
    beta = np.linalg.solve(X.T*X, X.T*F.R.Y)

    E = F.E.Y - F.E.X*beta
    rsq = rSqr(F.E.Y, E)

    # WHILE LOOP HERE
    # print('N train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0], '\tLinear regression adj R2 (test) = ',rsq)

    it = 0
    while it < 2000:

        yHat, xList, hList, gList, zpList = initialize(g, X, fns, dfns)

        xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)

        dEdyhat = -2 * (F.R.Y- yHat)

        gList = gradComputerOne(gList, xList, zpList, dEdyhat)

        hList[0] -= np.multiply(gamma, gList[0])

        obsAcc = testAcc(F.E, hList, fns)
        objFunction = sum([0.25* np.mean([x**2 for x in dEdyhat[:,i]]) for i in range(s)])
        obsAcc.append(objFunction)
        it += 1

    try:
        progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
    except(NameError):
        progress = obsAcc

    string = '\r'+str(k) + '/' + str(K) + ' \t' + str(it)
    for j in range(s):
        string += '\t r-sqr = '+ str(round(progress[j], 3))

    string +='\t obj = '+ str(round(progress[len(progress)-1], 5))
    print(string, end="")


    # print(testAcc(D, hList, fns))
