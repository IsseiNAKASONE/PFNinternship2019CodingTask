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


#dirpath = '../datasets/train/'
dirpath = './train/'

model = gnn.GNN()
data = D.TupleDataset(dirpath)
train_iter = Iterator(data, batch_size=1000)
optimizer = op.SGD()
optimizer.setup(model)
trainer = gnn.TrainGNN(train_iter, optimizer)
trainer.start(epoch=100)

