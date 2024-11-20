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
file_path = "./../datasets/CA-HepTh.txt"

# Load the graph
G = load_graph(file_path)

# Compute and print network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")


############## Task 2

import networkx as nx

# Assuming `G` is already loaded from Task 1

# Task 2: Number of connected components
num_components = nx.number_connected_components(G)
print(f"Number of connected components: {num_components}")

# Extract the largest connected component
largest_cc = max(nx.connected_components(G), key=len)  # Largest component by size
G_largest_cc = G.subgraph(largest_cc).copy()  # Subgraph of the largest component

# Statistics of the largest connected component
num_nodes_largest_cc = G_largest_cc.number_of_nodes()
num_edges_largest_cc = G_largest_cc.number_of_edges()
print(f"Number of nodes in the largest connected component: {num_nodes_largest_cc}")
print(f"Number of edges in the largest connected component: {num_edges_largest_cc}")

# Fractions of nodes and edges
node_fraction = num_nodes_largest_cc / G.number_of_nodes()
edge_fraction = num_edges_largest_cc / G.number_of_edges()
print(f"Fraction of nodes in the largest connected component: {node_fraction:.4f}")
print(f"Fraction of edges in the largest connected component: {edge_fraction:.4f}")

