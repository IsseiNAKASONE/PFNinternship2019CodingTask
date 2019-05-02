import fnmatch
import os
from random import random
import numpy as np
import functions as F



class TupleDataset:

    def __init__(self, filelist):
        self._graphs = [f[0] for f in filelist]
        self._labels = [f[1] for f in filelist]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [([read_graph(g), read_label(l)]) 
                    for g, l in zip(self._graphs[index], self._labels[index])] 
        else:
            return ([read_graph(self._graphs[index]),
                read_label(self._labels[index])])

    def __len__(self):
        return len(self._graphs)



def read_graph(filepath):
    with open(filepath) as fd:
        dim = int(fd.readline().split()[0])
        Adj = np.zeros((dim, dim))
        row = fd.readline().split()
        for i in range(dim):
            Adj[i:] = np.array(row)
            row = fd.readline().split()
    return Adj


def read_label(filepath):
    with open(filepath) as fd:
        return int(fd.readline().split()[0])


def get_dataset(dirname, test_ratio=0.25):
    train_filelist = []
    test_filelist = []

    names = os.listdir(dirname)
    graphs = sorted(fnmatch.filter(names, '*_graph.txt'))
    labels = sorted(fnmatch.filter(names, '*_label.txt'))

    for g, l in zip(graphs, labels):
        if random() > test_ratio:
            train_filelist.append((dirname+g, dirname+l))
        else:
            test_filelist.append((dirname+g, dirname+l))

    train_datasets = TupleDataset(train_filelist)
    test_datasets = TupleDataset(test_filelist)

    return train_datasets, test_datasets

