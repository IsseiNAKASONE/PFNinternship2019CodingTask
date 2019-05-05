import numpy as np
import datasets as D

   
def softplus(x):
    return np.maximum(0,x)+np.log(1+np.exp(-np.abs(x)))


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-709,100000)))


def heaviside(x, c=1/2):
    if not x:
        return c
    else:
        return 1 if x>0 else 0


def initial_vector(row, column):
    z = np.zeros((row, column)) 
    z[0, :] = 1
    return z


class BinaryCrossEntropy:

    def __init__(self, model, T, inputs, eps=0.001):
        self.model = model
        self.T = T
        self.graph = inputs[0]
        self.label = inputs[1]
        self.eps = eps
        self.forward()

    def __add__(self, bce):
        return self.data+self.bce

    def forward(self):
        self.h_G, self.s = self.model.forward(self.graph, self.T)

        ### calculate TPTN and loss
        if self.label:
            loss = softplus(-self.s)
            self.TPTN = 1 if self.s > 0 else 0
        else:
            loss = softplus(self.s)
            self.TPTN = 0 if self.s > 0 else 1
        self.data = loss

    def backward(self):
        model = self.model
        dim = self.model.dim 

        ### W
        dW = np.zeros((dim, dim))
        for r in range(dim):
            for c in range(dim):
                model_h = self.model.copy_model()
                model_h.W[r, c] += self.eps
                h_G_h, _ = model_h.forward(self.graph, self.T)
                dW[r, c] = np.dot(model.A, (h_G_h-self.h_G)/self.eps)

        ### A and b
        L_s = sigmoid(self.s)-self.label
        dW = L_s*dW
        dA = L_s*self.h_G.T

        return dW, dA, L_s

