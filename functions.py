import numpy as np
import datasets as D

   
def softplus(x):
    return np.maximum(0,x)+np.log2(1+np.exp(-np.abs(x)))


def sigmoid(x):
    return 1/(1+np.exp(-x))


def one_hot(dim):
    z = np.zeros(dim)
    z[0] = 1.
    return z


class binary_cross_entropy:

    def __init__(self, model, T, inputs, eps=0.001):
        self.model = model
        self.T = T
        self.graph = inputs[0]
        self.label = inputs[1]
        self.eps = eps
        self.data = 0. 

    def __add__(self, bce):
        return self.data+self.bce

    def forward(self, model=None):
        if model == None: model = self.model

        h_G = model(self.graph, self.T)
        s = np.dot(model.A, h_G)+model.b
        if self.label: loss = softplus(-s)
        else:          loss = softplus(s)
    
        return h_G, loss

    def backward(self):
        model = self.model
        dim = self.model.dim 
        h_G, L = self.forward()
        self.data = L

        ### W
        dW = np.zeros((dim, dim))
        for r in range(dim):
            for c in range(dim):
                de = np.zeros((dim, dim))
                de[r, c] += self.eps
                model_h = self.model.copy_model()
                model_h.W += de

                _, L_h = self.forward(model=model_h)
                dW[r, c] = (L_h-L)/self.eps

        ### A and b
        s = np.dot(model.A, h_G)+model.b
        L_s = sigmoid(s)-self.label
        dA = L_s*h_G.T

        return dW, dA, L_s
    
