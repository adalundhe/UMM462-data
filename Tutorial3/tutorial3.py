"""
    Sean Corbett
    04/18/2018
    M462
    Tutorial 1: Corrected
"""

from functions import *
from functions_2 import *
from random import choice, seed
from neural_network import ActivationFunction
from activation_functions import (
    rlu,
    rluPrime,
    softPlus,
    dsoftPlus
)

path = './data/parkinsons_updrs.csv'
path_boston = './data/bostonHousingData.txt'
path_breast_cancer = './data/breast-cancer-wisconsin.txt'


D_BreastCancer = breastCancerData(path_breast_cancer)
D_Boston = BostonHousing(path_boston)
D_Parkinsons = parkinsonsData(path)

loaded_datasets = [{'name': 'Breast Cancer Data' , 'data': D_BreastCancer},{'name': 'Boston Housing Data' , 'data': D_Boston}, {'name': 'Parkinsons Data' , 'data': D_Parkinsons}]

for D_dict in loaded_datasets:

    D = D_dict['data']

    n, p = dim(D.X)
    n, s = dim(D.Y)

    g = [p, p, p, s]
    K = 10

    z0 = ActivationFunction(rlu, rluPrime)
    z1 = ActivationFunction(rlu, rluPrime)
    z2 = ActivationFunction(softPlus, dsoftPlus)
    z3 = ActivationFunction(np.tanh, tanhP)

    fns = [z0.evaluate, z1.evaluate, z2.evaluate] if D_dict['name'] == 'Breast Cancer Data' else [z0.evaluate, z3.evaluate, z1.evaluate]
    dfns = [z0.differentiate, z1.differentiate, z2.differentiate] if D_dict['name'] == 'Breast Cancer Data' else [z0.differentiate, z3.differentiate, z1.differentiate]


    # z0 = ActivationFunction(rlu, rluPrime)
    # z1 = ActivationFunction(rlu, rluPrime)
    # z2 = ActivationFunction(identity, unit)
    # fns = [z0.evaluate, z1.evaluate, z2.evaluate]
    # dfns = [z0.differentiate, z1.differentiate, z2.differentiate]

    seed(0)
    np.random.seed(0)

    sampleID = [choice(range(K)) for i in range(n)]

    for k in range(K):
        # gamma = .00001

        if k == 1:
            break

        try:
            del progress
        except(NameError):
            pass

        F = getSample(D, sampleID, k)
        X = F.R.X
        beta = 0
        E = []
        rsq = 0


        if D.labels is None:
            beta = np.linalg.solve(X.T*X, X.T*F.R.Y)

            E = F.E.Y - F.E.X*beta
            rsq = rSqr(F.E.Y, E)

        # WHILE LOOP HERE
        # print('N train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0], '\tLinear regression adj R2 (test) = ',rsq)

        alphas = [0.001, 0.01, 0.1]
        it = 0
        yHat, xList, hList, gList, zpList, vList, iList, sgList = initialize(g, X, fns, dfns)

        for a in alphas:
            while it < 500:

                xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)
                dEdyhat = None

                if D.labels is None:
                    dEdyhat = -2 * (F.R.Y- yHat)
                else:
                    dEdyhat = -(np.multiply(F.R.Y, yHat)/np.multiply(yHat, (1-yHat)))

                alpha = .9 - .4*np.exp((1- it)*.001)

                m = len(hList)
                for r in range(m):
                    iList[r] = hList[r] - alpha*vList[r]

                gList = gradientComputerTwo(hList, gList, xList, zpList, dEdyhat)

                if D.labels is None:
                    for r in range(m):
                        sgList[r] = np.sqrt(.9*np.power(sgList[r], 2) + .1*np.power(gList[r], 2))
                        stepSizes = .01/sgList[r]
                        vList[r] = .9*vList[r] + np.multiply(stepSizes, gList[r])
                        hList[r] -= vList[r]
                else:
                    mag = a * np.exp(-it*a)
                    for r in range(m):
                        hList[r] -= mag*np.sign(gList[r])

                # hList[0] -= np.multiply(gamma, gList[0])

                obsAcc = testAcc(F.E, hList, fns)
                objFunction = None

                if D.labels is None:
                    objFunction = sum([0.25* np.mean([x**2 for x in dEdyhat[:,i]]) for i in range(s)])
                else:
                    objFunction = (-1.0/n) * np.sum(np.multiply(F.R.Y, np.log(yHat)) + np.multiply((1 - F.R.Y), np.log(1 - yHat)))
                obsAcc.append(objFunction)
                it += 1
                # print("ACC:",objFunction)

            try:
                progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
            except(NameError):
                progress = obsAcc

            string = '\r'+str(k) + '/' + str(K) + ' \t' + str(it)
            for j in range(s):
                string += '\t r-sqr = '+ str(round(progress[j], 3))

            string +='\t obj = '+ str(round(progress[len(progress)-1], 5)) + '\n'

            with open('results.txt', 'a') as results_file:
                results_file.write(D_dict['name']+": "+"\talpha: "+str(a)+"\n\nresults: "+string+'\n\n')

        with open('results.txt', 'a') as results_file:
            results_file.write(('-' * 10)+'FOLD: '+str(k+1)+' COMPLETE FOR DATASET: '+D_dict['name']+('-' * 10)+'\n\n\n')

    # print(string, end="")


    # print(testAcc(D, hList, fns))
