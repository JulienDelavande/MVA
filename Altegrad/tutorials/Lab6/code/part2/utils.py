"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    
    ##################
    num_graphs_per_class = 50
    
    # Classe 0
    for _ in range(num_graphs_per_class):
        n = randint(10, 20)
        G = nx.fast_gnp_random_graph(n, 0.2)
        Gs.append(G)
        y.append(0)

    # Classe 1
    for _ in range(num_graphs_per_class):
        n = randint(10, 20)
        G = nx.fast_gnp_random_graph(n, 0.4)
        Gs.append(G)
        y.append(1)
    ##################

    return Gs, y




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
