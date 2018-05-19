import numpy as np

class ActivationFunction(object):
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
    def evaluate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):
            for j in range(shape[1]):
                A[i,j] = self.function(X[i,j])

        return A

    def differentiate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):
            for j in range(shape[1]):
                A[i,j] = self.derivative(X[i,j])

        return A
