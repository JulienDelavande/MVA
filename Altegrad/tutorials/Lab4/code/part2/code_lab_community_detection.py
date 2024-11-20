"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans

def spectral_clustering(G, k):
    """
    Perform Spectral Clustering on a given graph.
    
    Parameters:
        G (networkx.Graph): Input graph.
        k (int): Number of clusters.
        
    Returns:
        dict: A dictionary mapping each node to its cluster.
    """
    # Step 1: Compute the adjacency matrix
    A = nx.adjacency_matrix(G).astype(float)  # Sparse matrix format

    # Step 2: Compute the degree matrix D
    degrees = np.array(A.sum(axis=1)).flatten()
    D = np.diag(degrees)
    
    # Step 3: Compute the normalized Laplacian matrix Lrw = I - D^-1 * A
    D_inv = np.diag(1 / degrees)  # Inverse of D
    Lrw = np.eye(A.shape[0]) - D_inv @ A.toarray()
    
    # Step 4: Compute the eigenvectors corresponding to the k smallest eigenvalues of Lrw
    eigenvalues, eigenvectors = np.linalg.eigh(Lrw)  # Symmetric matrix
    U = eigenvectors[:, :k]  # Take the first k eigenvectors
    
    # Step 5: Apply k-means clustering to the rows of U
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(U)
    labels = kmeans.labels_
    
    # Step 6: Return the cluster assignments
    clustering_result = {node: labels[i] for i, node in enumerate(G.nodes())}
    return clustering_result



############## Task 4

##################
# Load the graph (adjust the file path as needed)
file_path = "./datasets/CA-HepTh.txt"  # Replace with the actual file path
G = nx.read_edgelist(
    file_path, 
    delimiter='\t', 
    comments='#', 
    create_using=nx.Graph()
)

# Step 1: Extract the giant connected component
largest_cc = max(nx.connected_components(G), key=len)
G_largest_cc = G.subgraph(largest_cc).copy()

# Step 2: Apply Spectral Clustering on the giant component
k = 50  # Number of clusters
clusters = spectral_clustering(G_largest_cc, k)

# Step 3: Analyze the cluster assignments
# Count the number of nodes in each cluster
cluster_counts = {}
for cluster in clusters.values():
    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

# Print cluster statistics
print(f"Number of nodes in the largest connected component: {G_largest_cc.number_of_nodes()}")
print(f"Number of edges in the largest connected component: {G_largest_cc.number_of_edges()}")
print(f"Number of clusters: {k}")
print("Cluster sizes:")
for cluster_id, size in cluster_counts.items():
    print(f"Cluster {cluster_id}: {size} nodes")
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    """
    Compute the modularity of a graph clustering.

    Parameters:
        G (networkx.Graph): The input graph.
        clustering (dict): A dictionary mapping each node to its cluster ID.
        
    Returns:
        float: The modularity score.
    """
    # Step 1: Initialize variables
    m = G.number_of_edges()  # Total number of edges in the graph
    modularity = 0.0  # Initialize modularity score
    
    # Step 2: Group nodes by clusters
    clusters = {}
    for node, cluster_id in clustering.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    # Step 3: Compute modularity for each cluster
    for cluster_id, nodes in clusters.items():
        # Compute l_c: Number of edges within the cluster
        subgraph = G.subgraph(nodes)
        l_c = subgraph.number_of_edges()
        
        # Compute d_c: Sum of degrees of all nodes in the cluster
        d_c = sum(dict(G.degree(nodes)).values())
        
        # Add cluster's modularity contribution
        modularity += (l_c / m) - (d_c / (2 * m))**2
    
    return modularity

    
    
    
    
    return modularity



############## Task 6

##################
import random
import networkx as nx

# Load the graph (adjust file path as needed)
file_path = "CA-HepTh.txt"
G = nx.read_edgelist(
    file_path, 
    delimiter='\t', 
    comments='#', 
    create_using=nx.Graph()
)

# Extract the giant connected component
largest_cc = max(nx.connected_components(G), key=len)
G_largest_cc = G.subgraph(largest_cc).copy()

# Step 1: Spectral Clustering
k = 50
spectral_clusters = spectral_clustering(G_largest_cc, k)

# Compute modularity for Spectral Clustering
spectral_modularity = modularity(G_largest_cc, spectral_clusters)
print(f"Modularity for Spectral Clustering: {spectral_modularity}")

# Step 2: Random Partitioning
random_clusters = {node: random.randint(0, k-1) for node in G_largest_cc.nodes()}

# Compute modularity for Random Partitioning
random_modularity = modularity(G_largest_cc, random_clusters)
print(f"Modularity for Random Partitioning: {random_modularity}")

##################







