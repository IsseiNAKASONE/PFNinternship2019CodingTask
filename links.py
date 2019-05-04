import numpy as np



class Linear:

    def __init__(self, in_size, out_size, initialW=None):
        self._W =  np.random.normal(0, 0.4, (in_size, in_size)) if initialW is None else initialW.copy()
    
    def __call__(self, inputs):
        return np.dot(self.W, inputs)

    def copy(self):
        return self._W





class Chain:

    def __init__(self, *args):
        pass



