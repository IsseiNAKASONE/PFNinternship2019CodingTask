import numpy as np
import functions as F
import optimizers as op



class BinaryCrossEntropyMLP(F.BinaryCrossEntropy):

    def __init__(self, model, T, inputs, eps=0.001):
        super().__init__(model, T, inputs, eps)

    def backward(self):
        model = self.model
        dim = self.model.dim 

        ### W1 and W2
        dW1 = np.zeros((dim, 2*dim))
        for r in range(dim):
            for c in range(2*dim):
                model_h = self.model.copy_model()
                model_h.W1[r, c] += self.eps
                h_G_h, _ = model_h.forward(self.graph, self.T)
                dW1[r, c] = np.dot(model.A, (h_G_h-self.h_G)/self.eps)

        dW2 = np.zeros((2*dim, dim))
        for r in range(2*dim):
            for c in range(dim):
                model_h = self.model.copy_model()
                model_h.W2[r, c] += self.eps
                h_G_h, _ = model_h.forward(self.graph, self.T)
                dW2[r, c] = np.dot(model.A, (h_G_h-self.h_G)/self.eps)


        ### A and b
        L_s = F.sigmoid(self.s)-self.label
        dW1 = L_s*dW1
        dW2 = L_s*dW2
        dA = L_s*self.h_G.T

        return dW1, dW2, dA, L_s



class Adam(op.Adam):

    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, eps=10**-9):
        super().__init__(alpha, beta1, beta2, eps)
        self._m = tuple([0]*4)
        self._v = tuple([0]*4)

    def update(self, T, batch, lossfun=None):
        B = len(batch)
        grad = tuple([0]*4)
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

