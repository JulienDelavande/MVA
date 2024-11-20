"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
import networkx as nx

# Task 1: Load the network data
def load_graph(file_path):
    # Load the undirected graph, ignoring comment lines starting with "#"
    G = nx.read_edgelist(
        file_path, 
        delimiter='\t', 
        comments='#', 
        create_using=nx.Graph()
    )
    return G

# File path to the data
file_path = "CA-HepTh.txt"

# Load the graph
G = load_graph(file_path)

# Compute and print network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")


############## Task 2



