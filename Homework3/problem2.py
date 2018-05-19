from newtons_method import NewtonApproximator

def f(x):
    return x**2 - 2

def f_prime(x):
    return 2 * x



best_approximation = NewtonApproximator(f, f_prime, [0,1,100], 1e-12).approximate()

print('Best Approximation: x^2 - 2')
print('Root:', best_approximation['root'])
print('Error:', best_approximation['error'])
