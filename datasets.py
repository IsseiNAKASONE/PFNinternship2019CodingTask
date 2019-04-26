import fnmatch
import os
import numpy as np
import functions as F



class TupleDataset(object):

    def __init__(self, dirname):
        self._dirpath = dirname
        self._graphs = get_graph(dirname)
        self._labels = get_label(dirname)
        self._length = len(self._graphs)
        if self._length != len(self._labels):
            raise ValueError

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [tuple([read_graph(self._dirpath+g), read_label(self._dirpath+l)]) 
                    for g, l in zip(self._graphs[index], self._labels[index])] 
        else:
            g = self._dirpath+self._graphs[index]
            l = self._dirpath+self._labels[index]
            return tuple([read_graph(g), read_label(l)])

    def __len__(self):
        return self._length



def get_graph(dirname):
    names = os.listdir(dirname)
    return sorted(fnmatch.filter(names, '*_graph.txt'))


def get_label(dirname):
    names = os.listdir(dirname)
    return sorted(fnmatch.filter(names, '*_label.txt'))


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

