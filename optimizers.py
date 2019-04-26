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
        return loss.data



class SGD(Optimizer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, lossfun=None, *args, **kwds):
        batch = kwds['batch']
        B = len(batch)
        d_params = tuple([0]*3)

        for b in batch:
            loss = lossfun(*args, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]

        d_params = tuple([d/B for d in d_params])
        params = self.model.params
        n_params = tuple([p - self.alpha*d
                for p, d in zip(params, d_params)])
        self.model.param_update(n_params)
        
        return loss.data



class MomentumSGD(Optimizer):

    def __init__(self, alpha=0.0001, eta=0.9):
        self.alpha = alpha
        self.eta = eta
        dim = self.model.dim
        self._w = tuple(np.zeros((dim, dim)), np.zeros(dim), 0)

    def update(self, lossfun=None, *args, **kwds):
        batch = kwds['batch']
        d_params = tuple([0]*3)

        for b in batch:
            loss = lossfun(*args, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]

        params = self.model.params
        n_params = tuple([p - self.alpha*d + self.eta*self._w
                for p, d in zip(params, d_params)])
        self._w = tuple([-self.alpha*d + self.eta*w
                for w, d in zip(self._w, d_params)])
        self.model.param_update(n_params)
        
        return loss.data

