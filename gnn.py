import numpy as np
import functions as F



class GNN:

    def __init__(self, in_size=8, initialW=None,
            initialA=None, initialb=None):
        self.in_size = in_size
        ### initialize parameter
        self.W =  np.random.normal(0, 0.4, (in_size, in_size)) if initialW is None else initialW.copy()
        self.A =  np.random.normal(0, 0.4, in_size) if initialA is None else initialA.copy()
        self.b =  0 if initialb is None else initialb

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



class TrainGNN:

    def __init__(self, optimizer, train_iter, test_iter=None):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.log = {'epoch':[], 'main/loss':[], 'main/accuracy':[]}

    def start(self, epoch=100, T=2):
        print('epoch\tmain/loss\tmain/accuracy', end='')
        if self.test_iter is not None:
            print('\tval/loss\tval/main/accuracy', end='')
        print()
        self.T = T

        train_iter = self.train_iter
        while train_iter.epoch < epoch:
            batch = next(train_iter)
            self.optimizer.update(T, batch, F.binary_cross_entropy)
            if train_iter.is_new_epoch: self.print_report()
    
    def print_report(self):
        op = self.optimizer
        loss = op.loss
        accu = op.accuracy
        op.clear()

        self.log['epoch'].append(self.train_iter.epoch)
        self.log['main/loss'].append(loss)
        self.log['main/accuracy'].append(accu)

        print('{}'.format(self.train_iter.epoch),
                '\t{:.6f}'.format(loss),
                '\t{:.6f}'.format(accu), end='')

        if self.test_iter is not None:
            val_loss, val_accu = self.evaluate()
            print('\t{:.6f}'.format(val_loss),
                    '\t{:.6f}'.format(val_accu), end='')
        print()

    def evaluate(self):
        _val_loss = []
        _val_TPTN = []

        test_iter = self.test_iter
        model = self.optimizer.model

        while test_iter.epoch == 0:
            batch = next(test_iter)
            for b in batch:
                loss = F.binary_cross_entropy(model, self.T, b)
                _val_loss.append(loss.val_data)
                _val_TPTN.append(loss.val_TPTN)
        test_iter.reset()

        return np.average(np.array(_val_loss)), np.average(np.array(_val_TPTN)) 

