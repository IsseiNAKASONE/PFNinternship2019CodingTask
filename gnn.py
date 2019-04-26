import numpy as np
import functions as F
#import optimizers as op



class GNN(object):

    def __init__(self, in_size=8):
        self.X = F.one_hot(in_size)
        ### initialize parameter
        self.W = np.random.normal(0, 0.4, (in_size, in_size))
        self.A = np.random.normal(0, 0.4, in_size)
        self.b = 0
    
    def __call__(self, graph, T):
        X_t = self.X.copy()
        for t in range(T):
            h = np.dot(X_t, graph)
            X_t = np.maximum(0, np.dot(self.W, h))
        return X_t.sum(axis=1)

    def param_update(self):
        pass
        


class TrainGNN(object):

    def __init__(self, model, train, optimizer):
        self.model = model
        self.train = train
        self.optimizer = optimizer

    def start(self, epoch=100, batch_size=32, t=2):
        for e in range(epoch):
            print('epoch:', e+1, end='')
            W, A, b = F.gradient(self.model, self.X, self.label, t)
            self.model.W -= self.alpha*W
            self.model.A -= self.alpha*A
            self.model.b -= self.alpha*b

