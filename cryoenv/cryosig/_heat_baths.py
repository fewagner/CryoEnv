import numpy as np


class RandomWalkBath:
    """
    TODO
    
    $ dT = -k T + \sqrt{2 B} dW $

    Expectation <x> = 0
    
    Varianz $ <x^2> = \frac{B}{k} \left( 1 - \exp \left( - 2kt \right) \right)$
        
    or Varianz $ <x^2> = 2  B t $ if k = 0
        
    Equilibrium after $ \frac{1}{2k} $
    
    """

    def __init__(self, bath_temp=11., start_time=0., b=1e-6, k=1e-2):
        self.bath_temp = bath_temp
        self.temp = np.random.normal(loc=bath_temp, scale=np.sqrt(b/k))
        self.time_passed = start_time
        self.b = b
        self.k = k

    def __call__(self, t):
        if t > self.time_passed:  # if time passed
            self.temp -= self.k * (self.temp - self.bath_temp)
            self.temp += np.sqrt(2 * self.b) * np.random.normal(scale=t - self.time_passed)
            self.time_passed = t
        return self.temp
