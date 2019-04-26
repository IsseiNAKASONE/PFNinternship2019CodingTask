import fnmatch
import os
import numpy as np
import functions as F



class TupleDataset(object):

    def __init__(self, dirname):
        self._graphs = get_graph(dirname)
        self._labels = get_label(dirname)
        self._length = len(self._graphs)
        if self._length != len(self._labels):
            raise ValueError

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [tuple([graph, label]) 
                    for graph, label in zip(self._graphs[index], self._labels[index])] 
        else:
            return tuple([self._graphs[index], self._labels[index]])

    def __len__(self):
        return self._length



def get_graph(dirname):
    names = os.listdir(dirname)
    return sorted(fnmatch.filter(names, '*_graph.txt'))


def get_label(dirname):
    names = os.listdir(dirname)
    return sorted(fnmatch.filter(names, '*_label.txt'))


'''
        with open(filename) as fd:
            dim = int(fd.readline().split()[0])
            Adj = np.zeros((dim, dim))
            row = fd.readline().split()
            for i in range(dim):
                Adj[i:] = np.array(row)
                row = fd.readline().split()
        return Adj
'''

