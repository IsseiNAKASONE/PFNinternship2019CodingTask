import gnn
import functions as F



class Optimizer(object):
    model = None
    epoch = 0

    def setup(self, model):
        self.model = model
        self.epoch = 0
        return self

    def update(self, lossfun=None, *args):
        raise NotImplementedError



class GradientMethod(Optimizer):

    def __init__(self, eps=0.001):
        self.eps = eps

    def update(self, lossfun=None, *args):
        loss = lossfun(*args)
        

