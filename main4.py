'''
    PFN internship 2019 coding task
    machine learning
    task-4
    Issei NAKASONE
'''


import datasets as D
import mlp 
from gin import GIN, TrainGIN 
from iterator import Iterator


dirpath = '../datasets/train/'
predict = '../datasets/test/'
batch_size = 256

train = D.get_dataset(dirpath)
train_iter = Iterator(train, batch_size)

model = GIN()
optimizer = mlp.Adam()
optimizer.setup(model)
trainer = TrainGIN(optimizer, train_iter)
trainer.start(epoch=100)

pred = D.GraphDataset(predict)
trainer.predict(pred)

