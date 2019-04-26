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

    def __init__(self, alpha=0.01):
        self.alpha = alpha 

    def update(self, lossfun=None, *args, **kwds):
        inputs = kwds['batch'][0]
        loss = lossfun(*args, inputs)
        d_params = loss.backward()
        params = self.model.params
        n_params = tuple([p - self.alpha*d 
            for p, d in zip(params, d_params)])
        self.model.param_update(n_params)

