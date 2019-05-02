import numpy as np
import gnn
import functions as F



class Optimizer:
    model = None
    _loss = np.array([]) 
    _TPTN = np.array([]) 

    def __getattr__(self, key):
        if key == 'accuracy' and self._TPTN.size != 0:
            return np.average(self._TPTN)
        elif key == 'loss' and self._loss.size != 0:
            return np.average(self._loss)
        else:
            return None

    def setup(self, model):
        self.model = model
        self.reset()
        return self

    def update(self, T, batch, lossfun=None):
        raise NotImplementedError

    def report(self, loss, TPTN):
        self._loss = np.hstack((self._loss, np.array(loss)))
        self._TPTN = np.hstack((self._TPTN, np.array(TPTN)))

    def clear(self):
        self._loss = np.array([])
        self._TPTN = np.array([])

    def reset(self):
        pass



class GradientMethod(Optimizer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha 

    def update(self, T, batch, lossfun=None):
        inputs = batch[0]
        loss = lossfun(self.model, T, inputs)

        d_params = loss.backward()
        params = self.model.params
        n_params = tuple([p - self.alpha*d 
            for p, d in zip(params, d_params)])
        self.model.param_update(n_params)

        self.report(loss.data, loss.TPTN)


class SGD(Optimizer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        d_params = tuple([0]*3)
        loss_lst = []
        tptn = [] 

        for b in batch:
            loss = lossfun(self.model, T, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]
            loss_lst.append(loss.data)
            tptn.append(loss.TPTN)

        d_params = tuple([d/B for d in d_params])
        params = self.model.params
        n_params = tuple([p - self.alpha*d
                for p, d in zip(params, d_params)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, tptn)



class MomentumSGD(Optimizer):

    def __init__(self, alpha=0.01, eta=0.9):
        self.alpha = alpha
        self.eta = eta

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        d_params = tuple([0]*3)
        loss_lst = []
        tptn = []

        for b in batch:
            loss = lossfun(self.model, T, b)
            d_params = [d+l for d, l in zip(d_params, loss.backward())]
            loss_lst.append(loss.data)
            tptn.append(loss.TPTN)

        params = self.model.params
        n_params = tuple([p - self.alpha*d + self.eta*w
                for p, d, w in zip(params, d_params, self._w)])
        self._w = tuple([-self.alpha*d + self.eta*w
                for d, w in zip(d_params, self._w)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, tptn)
    
    def reset(self):
        dim = self.model.dim
        self._w = tuple([np.zeros((dim, dim)), np.zeros(dim), 0])

