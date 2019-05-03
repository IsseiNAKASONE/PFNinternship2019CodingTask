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


dirpath = '../datasets/train/'
batch_size = 128

train, test = D.get_dataset(dirpath, test_ratio=0.25)
train_iter = Iterator(train, batch_size)
test_iter = Iterator(test, batch_size)

model = gnn.GNN()
optimizer = op.SGD()
#optimizer = op.MomentumSGD()
optimizer.setup(model)
trainer = gnn.TrainGNN(optimizer, train_iter, test_iter)
trainer.start(epoch=50)

