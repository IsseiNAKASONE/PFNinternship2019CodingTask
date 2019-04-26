import numpy as np
import functions as F
#import optimizers as op



class GNN(object):

    def __init__(self, in_size=8, initialW=None,
            initialA=None, initialb=None):
        self.in_size = in_size
        ### initialize parameter
        if initialW is None:
            self.W = np.random.normal(0, 0.4, (in_size, in_size))
        else:
            self.W = initialW.copy()

        if initialA is None:
            self.A = np.random.normal(0, 0.4, in_size)
        else:
            self.A = initialA.copy()
        
        if initialb is None:
            self.b = 0
        else:
            self.b = initialb
    
    def __call__(self, graph, T):
        X = np.zeros((self.in_size, graph.shape[0])) 
        X[0, :] = 1
        for t in range(T):
            h = np.dot(X, graph)
            X = np.maximum(0, np.dot(self.W, h))
        return X.sum(axis=1)

    def __getattr__(self, key):
        if key == 'dim':      return self.in_size
        elif key == 'params': return self.W, self.A, self.b
        else: raise AttributeError

    def param_update(self, params):
        self.W = params[0]
        self.A = params[1]
        self.b = params[2]

    def copy_model(self):
        return GNN(self.in_size, self.W, self.A, self.b)



class TrainGNN(object):

    def __init__(self, train_iter, optimizer):
        self.train_iter = train_iter
        self.optimizer = optimizer

    def start(self, epoch=100, T=2):
        train_iter = self.train_iter

        while train_iter.epoch < epoch:
            batch = next(train_iter)
            model = self.optimizer.model
            loss = self.optimizer.update(F.binary_cross_entropy, model, T, batch=batch)
            
            if train_iter.is_new_epoch: 
                print('epoch:', self.train_iter.epoch, '\tloss:', loss)

