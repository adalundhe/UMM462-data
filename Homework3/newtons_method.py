import numpy as np

class NewtonApproximator():
    def __init__(self, f, f_prime, range_settings, error_tolerance):
        self.f = f
        self.f_prime = f_prime
        self.range = np.linspace(range_settings[0], range_settings[1] , range_settings[2])
        self.error_tolerance = error_tolerance

    def delta_x(self, x):
        return abs(0-self.f(x))

    def newtons_method(self, x0):
        if x0 == 0:
            return 0

        delta = self.delta_x(x0)
        while delta > self.error_tolerance:
            x0 = x0 - self.f(x0)/self.f_prime(x0)
            delta = self.delta_x(x0)

        return {'root': x0, 'error': self.f(x0)}

    def approximate(self):
        results = [self.newtons_method(point) for point in self.range]

        return min(results[1:], key=lambda x: x['root'])
