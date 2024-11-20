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
file_path = "CA-HepTh.txt"  # Replace with the actual file path
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
    
    ##################
    # your code here #
    ##################
    
    
    
    
    return modularity



############## Task 6

##################
# your code here #
##################







