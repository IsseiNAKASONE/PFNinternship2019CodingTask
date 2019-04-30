import numpy as np
import gnn
import functions as F



class Optimizer:
    model = None
    _loss = np.array([]) 
    _total = 0
    _TPTN = 0 

    def __getattr__(self, key):
        if key == 'accuracy' and self._total != 0:
            return self._TPTN/self._total
        elif key == 'loss' and self._loss.size != 0:
            return np.average(self._loss)
        else:
            return None

    def setup(self, model):
        self.model = model
        self.reset()
        return self

    def update(self, lossfun=None, *args, **kwds):
        raise NotImplementedError

    def report(self, loss, total, TPTN):
        self._loss = np.hstack((self._loss, loss))
        self._total += total
        self._TPTN += TPTN

    def clear(self):
        self._loss = np.array([])
        self._total = 0
        self._TPTN = 0

    def reset(self):
        pass



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

        self.report(loss.data, 1, loss.TPTN)


class SGD(Optimizer):

    def __init__(self, alpha=0.0001):
        self.alpha = alpha

    def update(self, lossfun=None, *args, **kwds):
        B = len(kwds['batch'])
        d_params = tuple([0]*3)
        loss_lst = []
        tptn = 0

        for b in kwds['batch']:
            loss = lossfun(*args, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]
            loss_lst = np.hstack((loss_lst, loss.data))
            tptn += loss.TPTN

        d_params = tuple([d/B for d in d_params])
        params = self.model.params
        n_params = tuple([p - self.alpha*d
                for p, d in zip(params, d_params)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, B, tptn)



class MomentumSGD(Optimizer):

    def __init__(self, alpha=0.0001, eta=0.9):
        self.alpha = alpha
        self.eta = eta

    def update(self, lossfun=None, *args, **kwds):
        B = len(kwds['batch'])
        d_params = tuple([0]*3)
        loss_lst = np.array([])
        tptn = 0

        for b in kwds['batch']:
            loss = lossfun(*args, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]
            loss_lst = np.hstack((loss_lst, loss.data))
            tptn += loss.TPTN

        params = self.model.params
        n_params = tuple([p - self.alpha*d + self.eta*w
                for p, d, w in zip(params, d_params, self._w)])
        self._w = tuple([-self.alpha*d + self.eta*w
                for d, w in zip(d_params, self._w)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, B, tptn)
    
    def reset(self):
        dim = self.model.dim
        self._w =tuple([np.zeros((dim, dim)), np.zeros(dim), 0])

