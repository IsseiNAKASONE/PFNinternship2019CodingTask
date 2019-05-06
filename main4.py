'''
    PFN internship 2019 coding task
    machine learning
    task-4
    Issei NAKASONE
'''


import datasets as D
import optimizers as op
from gnn import GNN, TrainGNN 
from iterator import Iterator


dirpath = '../datasets/train/'
predict = '../datasets/test/'
batch_size = 256

train = D.get_dataset(dirpath)
train_iter = Iterator(train, batch_size)

model = GNN()
optimizer = op.Adam()
optimizer.setup(model)
trainer = TrainGNN(optimizer, train_iter)
trainer.start(epoch=100)

pred = D.GraphDataset(predict)
trainer.predict(pred)

