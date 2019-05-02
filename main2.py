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

filelist = [('../datasets/train/0_graph.txt',
    '../datasets/train/0_label.txt')]


model = gnn.GNN()
data = D.TupleDataset(filelist)
train_iter = Iterator(data, batch_size=1)
optimizer = op.GradientMethod()
optimizer.setup(model)
trainer = gnn.TrainGNN(optimizer, train_iter)
trainer.start(epoch=100, T=2)

