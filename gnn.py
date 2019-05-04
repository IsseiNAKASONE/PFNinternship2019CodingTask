import numpy as np
import functions as F
from reporter import Reporter



class GNN:

    def __init__(self, in_size=8, initialW=None,
            initialA=None, initialb=None):
        self.in_size = in_size
        ### initialize parameter
        self.W =  np.random.normal(0, 0.4, (in_size, in_size)) if initialW is None else initialW.copy()
        self._A =  np.random.normal(0, 0.4, in_size) if initialA is None else initialA.copy()
        self._b =  0 if initialb is None else initialb

    def __call__(self, graph, T):
        X = np.zeros((self.in_size, graph.shape[0])) 
        X[0, :] = 1
        for t in range(T):
            h = np.dot(X, graph)
            X = np.maximum(0, np.dot(self.W, h))
        h_G = X.sum(axis=1)
        
        s = np.dot(self._A, h_G)+self._b
        return h_G, s 

    def __getattr__(self, key):
        if key == 'dim':      return self.in_size
        elif key == 'params': return self.W, self._A, self._b
        else: raise AttributeError

    def param_update(self, params):
        self.W = params[0]
        self._A = params[1]
        self._b = params[2]

    def copy_model(self):
        return GNN(self.in_size, self.W, self._A, self._b)



class TrainGNN:

    def __init__(self, optimizer, train_iter, test_iter=None):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.reporter = Reporter() 

    def start(self, epoch=100, T=2):
        self.T = T

        train_iter = self.train_iter
        while train_iter.epoch < epoch:
            batch = next(train_iter)
            self.optimizer.update(T, batch, F.BinaryCrossEntropy)

            if train_iter.is_new_epoch:
                loss = self.optimizer.loss
                accu = self.optimizer.accuracy
                self.optimizer.clear()

                self.reporter.report('epoch', int(train_iter.epoch))
                self.reporter.report('main/loss', loss)
                self.reporter.report('main/accuracy', accu)

                if self.test_iter is not None: self.evaluate()
                self.reporter.print_report()

        self.reporter.log_report()
    
    def evaluate(self):
        _val_loss = []
        _val_TPTN = []

        test_iter = self.test_iter
        model = self.optimizer.model

        while test_iter.epoch == 0:
            batch = next(test_iter)
            for b in batch:
                loss = F.BinaryCrossEntropy(model, self.T, b)
                _val_loss.append(loss.val_data)
                _val_TPTN.append(loss.val_TPTN)
        test_iter.reset()

        self.reporter.report('test/main/loss', np.average(np.array(_val_loss)))
        self.reporter.report('test/main/accuracy', np.average(np.array(_val_TPTN)))

    def predict(self, pred, model=None, outfile='predict.txt'):
        if model is None: model = self.optimizer.model

        pred_list = []
        for p in pred:
            _, s = model(p, self.T)
            p = F.sigmoid(s)
            if p > 1/2:
                pred_list.append(1)
            else:
                pred_list.append(0)

        with open(outfile, 'w') as fd:
            for y in pred_list: fd.write(str(y)+'\n')

