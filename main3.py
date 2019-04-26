'''
    PFN internship 2019 coding task
    machine learning
    task-3
    Issei NAKASONE
'''

import gnn
import datasets as D
import optimizers as op
from iterator import Iterator


train_dir = '../datasets/train/'
#test_dir = '../datasets/test/'

model = gnn.GNN()
train= D.TupleDataset(dirpath)
train_iter = Iterator(train, batch_size=1)
optimizer = op.MomentumSGD()
optimizer.setup(model)
train_gnn = gnn.TrainGNN(train_iter, optimizer)
train_gnn.start()

