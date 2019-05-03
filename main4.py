'''
    PFN internship 2019 coding task
    machine learning
    task-4
    Issei NAKASONE
'''

import gnn
import datasets as D
import optimizers as op
from iterator import Iterator


dirpath = '../datasets/train/'
predict = '../datasets/test/'
batch_size = 128

train = D.get_dataset(dirpath)
train_iter = Iterator(train, batch_size)

model = gnn.GNN()
optimizer = op.Adam()
optimizer.setup(model)
trainer = gnn.TrainGNN(optimizer, train_iter)
trainer.start(epoch=100)

pred = D.GraphDataset(predict)
trainer.predict(pred)

