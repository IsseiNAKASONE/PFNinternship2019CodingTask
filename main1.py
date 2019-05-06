'''
    PFN internship 2019 coding task
    machine learning
    task-1
    Issei NAKASONE
'''

import datasets as D
from gnn import GNN

filepath = '../datasets/train/0_graph.txt'
graph = D.read_graph(filepath)
model = GNN()
h_G, _ = model.forward(graph, T=2)
print(h_G)

