import numpy as np
import functions as F
import mlp
from reporter import Reporter



class GIN:

    def __init__(self, in_size=8, initialW1=None, initialW2=None,
            initialA=None, initialb=None):
        self.in_size = in_size
        ### initialize parameter
        self.W1 =  np.random.normal(0, 0.4, (in_size, 2*in_size)) if initialW1 is None else initialW1.copy()
        self.W2 =  np.random.normal(0, 0.4, (2*in_size, in_size)) if initialW2 is None else initialW2.copy()
        self.A =  np.random.normal(0, 0.4, in_size) if initialA is None else initialA.copy()
        self.b =  0 if initialb is None else initialb

    def __call__(self, graph, T):
        _, s = self.forward(graph, T)
        p = F.sigmoid(s)
        if p > 1/2: return 1      
        else:       return 0

    def __getattr__(self, key):
        if key == 'dim':      return self.in_size
        elif key == 'params': return self.W1, self.W2, self.A, self.b
        else: raise AttributeError
    
    def forward(self, graph, T):
        X = F.initial_vector(self.in_size, graph.shape[0])
        for t in range(T):
            h = np.dot(X, graph+np.identity(graph.shape[0]))
            X = np.maximum(0, np.dot(self.W1.T, h))
            X = np.maximum(0, np.dot(self.W2.T, X))
        h_G = X.sum(axis=1)
        s = np.dot(self.A, h_G)+self.b
        return h_G, s

    def param_update(self, params):
        self.W1 = params[0]
        self.W2 = params[1]
        self.A = params[2]
        self.b = params[3]

    def copy_model(self):
        return GIN(self.in_size, self.W1, self.W2, self.A, self.b)



class TrainGIN:

    def __init__(self, optimizer, train_iter, test_iter=None):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.reporter = Reporter() 

    def start(self, epoch=100, T=2, outfile='log.json'):
        self.T = T

        train_iter = self.train_iter
        while train_iter.epoch < epoch:
            batch = next(train_iter)
            self.optimizer.update(T, batch, mlp.BinaryCrossEntropyMLP)

            if train_iter.is_new_epoch:
                loss = self.optimizer.loss
                accu = self.optimizer.accuracy
                self.optimizer.clear()

                self.reporter.report('epoch', int(train_iter.epoch))
                self.reporter.report('main/loss', loss)
                self.reporter.report('main/accuracy', accu)

                if self.test_iter is not None: self.evaluate()
                self.reporter.print_report()

        self.reporter.log_report(outfile)

    def evaluate(self):
        _val_loss = []
        _val_TPTN = []

        test_iter = self.test_iter
        model = self.optimizer.model

        while test_iter.epoch == 0:
            batch = next(test_iter)
            for b in batch:
                loss = F.BinaryCrossEntropy(model, self.T, b)
                _val_loss.append(loss.data)
                _val_TPTN.append(loss.TPTN)
        test_iter.reset()

        self.reporter.report('test/main/loss', np.average(np.array(_val_loss)))
        self.reporter.report('test/main/accuracy', np.average(np.array(_val_TPTN)))

    def predict(self, pred, outfile='prediction.txt'):
        model = self.optimizer.model
        pred_list = [model(p, self.T) for p in pred]
        with open(outfile, 'w') as fd:
            for y in pred_list: fd.write(str(y)+'\n')

