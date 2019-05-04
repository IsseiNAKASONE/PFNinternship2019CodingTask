import numpy as np
import datasets as D

   
def softplus(x):
    return np.maximum(0,x)+np.log(1+np.exp(-np.abs(x)))


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-709,100000)))


def one_hot(dim):
    z = np.zeros(dim)
    z[0] = 1.
    return z


class BinaryCrossEntropy:

    def __init__(self, model, T, inputs, eps=0.001):
        self.model = model
        self.T = T
        self.graph = inputs[0]
        self.label = inputs[1]
        self.eps = eps
        self.data = 0.
        self.TPTN = None

    def __add__(self, bce):
        return self.data+self.bce

    def __getattr__(self, key):
        if key == 'val_data':
            _, _, loss = self.forward()
            return loss
        if key == 'val_TPTN':
            if self.TPTN is None: self.forward()
            return self.TPTN

    def forward(self, model=None):
        if model is None: model = self.model
        h_G, s = model(self.graph, self.T)

        ### calculate TPTN and loss
        if self.label:
            loss = softplus(-s)
            if self.TPTN is None:
                self.TPTN = 1 if s > 0 else 0
        else:
            loss = softplus(s)
            if self.TPTN is None:
                self.TPTN = 0 if s > 0 else 1 
    
        return h_G, s, loss

    def backward(self):
        model = self.model
        dim = self.model.dim 
        h_G, s, L = self.forward()
        self.data = L

        ### W
        dW = np.zeros((dim, dim))
        for r in range(dim):
            for c in range(dim):
                model_h = self.model.copy_model()
                model_h.W[r, c] += self.eps
                _, _, L_h = self.forward(model=model_h)
                dW[r, c] = (L_h-L)/self.eps

        ### A and b
        L_s = sigmoid(s)-self.label
        dA = L_s*h_G.T

        return dW, dA, L_s

