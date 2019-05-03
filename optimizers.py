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

        grad = loss.backward()
        params = self.model.params
        n_params = tuple([p - self.alpha*d 
            for p, d in zip(params, grad)])
        self.model.param_update(n_params)

        self.report(loss.data, loss.TPTN)


class SGD(Optimizer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        grad = tuple([0]*3)
        loss_lst = []
        tptn = [] 

        for b in batch:
            loss = lossfun(self.model, T, b)
            grad = [d+l for d, l in zip(grad, loss.backward())]
            loss_lst.append(loss.data)
            tptn.append(loss.TPTN)
        grad = tuple([g/B for g in grad])

        params = self.model.params
        n_params = tuple([p - self.alpha*g
                for p, g in zip(params, grad)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, tptn)



class MomentumSGD(Optimizer):

    def __init__(self, alpha=0.0001, eta=0.9):
        self.alpha = alpha
        self.eta = eta
        self._w = tuple([0]*3)

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        grad = tuple([0]*3)
        loss_lst = []
        tptn = []

        for b in batch:
            loss = lossfun(self.model, T, b)
            grad = [d+l for d, l in zip(grad, loss.backward())]
            loss_lst.append(loss.data)
            tptn.append(loss.TPTN)
        grad = tuple([g/B for g in grad])

        params = self.model.params
        n_params = tuple([p - self.alpha*g + self.eta*w
                for p, g, w in zip(params, grad, self._w)])
        self._w = tuple([-self.alpha*d + self.eta*w
                for d, w in zip(grad, self._w)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, tptn)
    


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=10**-9):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = tuple([0]*3)
        self._v = tuple([0]*3)
        self._t = 0

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        grad = tuple([0]*3)
        loss_lst = []
        tptn = []

        for b in batch:
            loss = lossfun(self.model, T, b)
            grad = [d+l for d, l in zip(grad, loss.backward())]
            loss_lst.append(loss.data)
            tptn.append(loss.TPTN)
        grad = tuple([g/B for g in grad])
        
        self._t += 1
        self._m = tuple([self.beta1*m + (1-self.beta1)*g
            for m, g in zip(self._m, grad)])
        self._v = tuple([self.beta2*v + (1-self.beta2)*(g**2)
            for v, g in zip(self._v, grad)])
        m_hat = tuple([m/(1-self.beta1**self._t) for m in self._m])
        v_hat = tuple([v/(1-self.beta2**self._t) for v in self._v])

        params = self.model.params
        n_params = tuple([p - self.alpha*m/(np.sqrt(v)+self.eps)
                for p, m, v in zip(params, m_hat, v_hat)])
        self.model.param_update(n_params)
        
        self.report(loss_lst, tptn)

