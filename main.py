import gnn 
import datasets as D
import optimizers as op
from iterator import Iterator

batch_size = 32
train_dir = './train'
# test_dir = '../datasets/test'

### preprocessing
model = gnn.GNN()
train = D.TupleDataset(train_dir)
train_iter = Iterator(train, batch_size)

### task-1
# print(h_G)

### task-2
optimizer = op.GradientMethod() 
optimizer.setup(model)

