import numpy as np

def rlu(x):
    return max([0,x])

def rluPrime(x):
    return 1 if x > 0 else 0

def identity(x):
    return x

def unit(x):
    return 1

def softPlus(x):
    return 1/(1 + np.exp(-x))

def dsoftPlus(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)
