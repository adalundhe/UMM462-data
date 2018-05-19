"""
    Sean Corbett
    04/18/2018
    M462
    Tutorial 1: Corrected
"""

from functions import *
from functions_2 import *
from random import choice, seed, shuffle
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

    # Set the batch sample size
    batSize = 200
    # Begin K iterations of cross-validation:
    cvID = [choice(range(K)) for i in range(n)]
    objFn = None
    dEdyhatf = None

    for k in range(K):

        try:
            del progress
        except(NameError):
            pass

        C = getSample(D, cvID, k)
        n = dim(C.R.Y)[0] # cross-validation training set size:
        nBatSamples = int(n/batSize) if int(n/batSize) > 0 else 1 # number of batch samples
        print(nBatSamples)
        # Assign each CV training observation to a batch sample:
        batchID = [choice(range(nBatSamples)) for i in range(n)]
        B = getSample(C.R, batchID, 0) # Draw the first batch sample = B.E
        # Set the counters
        batSize = dim(B.E.X)[0]
        batProgress = 0
        nProcessed = 0

        it = 0
        initialList = initialize(g, B.E.X, fns, dfns)
        yHat, xList, hList, gList, zpList, vList, iList, sgList = initialList

        epoch = 0
        for it in range(10000):
            epoch = int(nProcessed/n)
            # Carry out back propagation as usual

            xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)


            if D.labels is None:
                # target is quantitative
                objFn = rmse
                dEdyhatf = dEdyhatSqr
            else:
                # target is qualitative
                objFn = crossEntropy
                dEdyhatf = dEdyhatCE

            dEdyhat = dEdyhatf(B.E.Y, yHat)

            alpha = .9 - .4*np.exp((1- it)*.001)

            m = len(hList)
            for r in range(m):
                iList[r] = hList[r] - alpha*vList[r]

            gList = gradientComputerTwo(iList, gList, xList, zpList, dEdyhat)

            if D.labels is None:
                for r in range(m):
                    sgList[r] = np.sqrt(.9*np.power(sgList[r], 2) + .1*np.power(gList[r], 2))
                    stepSizes = .01/sgList[r]
                    vList[r] = .9*vList[r] + np.multiply(stepSizes, gList[r])
                    hList[r] -= vList[r]
            else:
                mag = alpha * np.exp(-it*alpha)
                for r in range(m):
                    hList[r] -= mag*np.sign(gList[r])

            # hList[0] -= np.multiply(gamma, gList[0])

            obsAcc = testAcc(C.E, hList, fns)
            objFunction = objFn(B.E.Y, yHat)

            obsAcc.append(objFunction)
            # Back propagation steps complete
            # Determine number of observations used to date:
            batProgress += dim(B.E.X)[0]
            nProcessed += dim(B.E.X)[0]
            # Test is the epoch is complete.
            # If so, then create a new batch sample index
            if batProgress >= n:
                # shuffle the batch sample indexes in place
                shuffle(batchID)
                batProgress = 0
            # Determine the batch sample for the next iteration and get it
            label = (it+1)%nBatSamples
            B = getSample(C.R, batchID, label)
            # Input the new X.
            xList[0] = B.E.X
            # Return to the start of the back propagation for loop

            try:
                progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
            except(NameError):
                progress = obsAcc

        string = '\r'+str(k) + '/' + str(K) + ' \t' + str(it)
        for j in range(s):
            string += '\t r-sqr = '+ str(round(progress[j], 3))

        string +='\t obj = '+ str(round(progress[len(progress)-1], 5)) + '\n'

        with open('results.txt', 'a') as results_file:
            results_file.write(D_dict['name']+": "+"\talpha: "+str(alpha)+"\n\nresults: "+string+'\n\n')

    with open('results.txt', 'a') as results_file:
        results_file.write(('-' * 10)+'FOLD: '+str(k+1)+' COMPLETE FOR DATASET: '+D_dict['name']+('-' * 10)+'\n\n\n')
