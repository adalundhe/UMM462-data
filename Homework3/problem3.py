from newtons_method import NewtonApproximator
import numpy as np

def f(x):
    return np.exp(x) - x - 1

def f_prime(x):
    return np.exp(x) - 1



best_approximation = NewtonApproximator(f, f_prime, [0,1,100], 1e-20).approximate()

print('Best Approximation: e^x - x = 1')
print('Root:', best_approximation['root'])
print('Error:', best_approximation['error'])
