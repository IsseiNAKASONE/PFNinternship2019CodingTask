import numpy as np


def one_hot(dim):
    z = np.zeros(dim)
    z[0] = 1.
    return z


def binary_cross_entropy(model, X, y, t, aggr=True):
    if aggr: model(X, t) 
    s = np.dot(model.A, model.h_G)+model.b
    if y: return softplus(-s)
    else: return softplus(s)


def softplus(x):
    return np.maximum(0,x)+np.log(1+np.exp(-np.abs(x)))


def sigmoid(x):
    return 1/(1+np.exp(-x))


def gradient(model, X, y, t, eps=0.001, func=binary_cross_entropy):
    dim = model.W.shape[0]
    h_G = model.h_G
    L = func(model, X, y, t)
    print('\tloss:', L)
    
    ### W
    de = np.zeros((dim, dim))
    swp = np.zeros((dim, dim))
    dW = np.zeros((dim, dim))
    for r in range(dim):
        for c in range(dim):
            de[r,c] += eps
            swp = model.W.copy()
            model.W += de
            L_t = func(model, X, y, t)

            model.W = swp.copy()
            dW[r,c]=(L_t-L)/eps
            de[r,c] = 0
    
    ### A and b
    s = np.dot(model.A, h_G)+model.b
    L_s = sigmoid(s)-y
    dA = L_s*h_G.T

    return dW, dA, L_s 
    
