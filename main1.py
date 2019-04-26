'''
    PFN internship 2019 coding task
    machine learning
    task-1
    Issei NAKASONE
'''

import gnn
import datasets as D

filepath = 'mono/0_graph.txt'
model = gnn.GNN()
graph = D.read_graph(filepath)
h_G = model(graph, T=2)
print(h_G)

