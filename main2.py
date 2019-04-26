'''
    PFN internship 2019 coding task
    machine learning
    task-2
    Issei NAKASONE
'''

import gnn
import datasets as D
import optimizers as op
from iterator import Iterator


dirpath = './mono/'

model = gnn.GNN()
data = D.TupleDataset(dirpath)
train_iter = Iterator(data, batch_size=1)
optimizer = op.GradientMethod()
optimizer.setup(model)
trainer = gnn.TrainGNN(train_iter, optimizer)
trainer.start(epoch=100, T=2)

