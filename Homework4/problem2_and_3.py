from exoplanetsReducer import reduceData, dim
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from random import choice, randint

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

def run_regression(X, Y, epsilon, n, q):
    """
        Runs logistic regression given X, Y, and epsilon using log-likelihood.
    """
    count = 1
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

def estimated_probabilities(X, beta, thresholds):
    probs = [0] * dim(X)[0]
    for i,threshold in enumerate(thresholds):
        probs[i] = [sigmoid(np.dot(X, beta)) >= threshold]

    return probs

def calc_confusion_mtx(y, y_hats):
    specificities, sensitivities = [0]*dim(y)[0], [0]*dim(y)[0]

    for i, y_h in enumerate(y_hats):
        n_00 = dim(np.where(np.logical_and(y==0, y_h[0]==0) == 1)[0])[0]
        n_01 = dim(np.where(np.logical_and(y==0, y_h[0]==1) == 1)[0])[0]
        n_10 = dim(np.where(np.logical_and(y==1, y_h[0]==0) == 1)[0])[0]
        n_11 = dim(np.where(np.logical_and(y==1, y_h[0]==1) == 1)[0])[0]

        n_0_plus = n_00+ n_01
        n_1_plus = n_10 + n_11

        sensitivity = n_11/n_1_plus
        specificity = n_00/n_0_plus

        sensitivities[i] = sensitivity
        specificities[i] = specificity

    return sensitivities, specificities

def k_fold(k,n, X, Y):
    for j in range(k):
        sampleID = [choice(range(k)) for i in range(n)]
        sIndex = [i for i in range(n) if sampleID[i] == j]
        x_train = X[sIndex]
        y_train = Y[sIndex]

        n, q = dim(x_train)
        beta = run_regression(x_train, y_train, 1e-4, n, q)
        # piHat.extend([1/(1 + np.exp(-x*b)) for x in D.X[sIndex,:]])

        thresholds = np.linspace(.001, .25, n)
        predicted = estimated_probabilities(x_subset[sIndex,:], beta, thresholds)
        sensitivities, specificities = calc_confusion_mtx(y_subset[sIndex,:], predicted)

        plt.plot(thresholds, sensitivities)
        plt.xlabel('Threshold')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity vs Threshold - K-Fold: {}'.format(j))
        plt.show()

        plt.plot(thresholds, specificities)
        plt.xlabel('Threshold')
        plt.ylabel('Specificity')
        plt.title('Specificity vs Threshold K-Fold: {}'.format(j))
        plt.show()

def maximize_sensitivity(target, x_data, y_data, max_threshold, drop_percentage):
    n, q = dim(x_data)
    thresholds = np.linspace(max_threshold * drop_percentage, max_threshold, n)
    sensitivities = np.asarray([0] * n)
    n_iters = 0

    while sensitivities[0] < target:
        beta = run_regression(x_subset, y_subset, 1e-5, n , q)
        predicted = estimated_probabilities(x_subset, beta, thresholds)
        sensitivities, specificities = calc_confusion_mtx(y_subset, predicted)

        thresholds = np.linspace(thresholds[0] * drop_percentage, thresholds[0], n)

        n_iters += 1

    return thresholds, sensitivities, specificities, n_iters

path = './data/exoTrain.csv'
D = reduceData(path)

X = D.X
Y = D.Y
n, q = dim(D.X)
n_1, q_1 = dim(D.Y)
slice_size = 1500

x_subset = X[:slice_size]
y_subset = Y[:slice_size]

# k_fold(3, slice_size, x_subset, y_subset)
#
# beta = run_regression(x_subset, y_subset, 1e-5, n , q)
#
# thresholds = np.linspace(.001, .25, slice_size)
# predicted = estimated_probabilities(x_subset, beta, thresholds)
#
# sensitivities, specificities = calc_confusion_mtx(y_subset, predicted)

maxed_thresholds, maxed_sensitivities, specificities, convergence_iterations = maximize_sensitivity(0.8, x_subset, y_subset, 0.2, 0.8)

print("Converged in:",convergence_iterations,"iterations.")

# plt.plot(thresholds, sensitivities)
# plt.xlabel('Threshold')
# plt.ylabel('Sensitivity')
# plt.title('Sensitivity vs Threshold')
# plt.show()
#
# plt.plot(thresholds, specificities)
# plt.xlabel('Threshold')
# plt.ylabel('Specificity')
# plt.title('Specificity vs Threshold')
# plt.show()

plt.plot(maxed_thresholds, maxed_sensitivities)
plt.xlabel('Maxed Threshold')
plt.ylabel('Maxed Sensitivity')
plt.title('Maxed Sensitivity vs Maxed Threshold')
plt.show()

plt.plot(maxed_thresholds, specificities)
plt.xlabel('Maxed Threshold')
plt.ylabel('Specificity')
plt.title('Specificity vs Maxed Threshold')
plt.show()
