"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt


# Loads the karate network
G = nx.read_weighted_edgelist('./data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
color_map = ['blue' if label == 0 else 'red' for label in y]

# Tracer le réseau avec NetworkX
plt.figure(figsize=(10, 10))
nx.draw_networkx(
    G,
    with_labels=True,
    node_color=color_map,
    node_size=500,
    font_size=10,
    font_color='white'
)

plt.title('Visualization of the Karate Network', fontsize=20)
plt.savefig('karate_network.pdf')
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Classification Accuracy: {accuracy:.4f}")
##################


############## Task 8
# Generates spectral embeddings

##################
A = nx.adjacency_matrix(G).astype(float)
degrees = np.array(A.sum(axis=1)).flatten()
D_inv = np.diag(1 / degrees)
Lrw = np.eye(A.shape[0]) - D_inv @ A.toarray()

eigenvalues, eigenvectors = np.linalg.eigh(Lrw)
spectral_embeddings = eigenvectors[:, :2]

X_train_spectral = spectral_embeddings[idx_train, :]
X_test_spectral = spectral_embeddings[idx_test, :]

classifier_spectral = LogisticRegression(random_state=42, max_iter=1000)
classifier_spectral.fit(X_train_spectral, y_train)

y_pred_spectral = classifier_spectral.predict(X_test_spectral)
accuracy_spectral = accuracy_score(y_test, y_pred_spectral)

print(f"DeepWalk Classification Accuracy: {accuracy:.4f}")
print(f"Spectral Embeddings Classification Accuracy: {accuracy_spectral:.4f}")
##################
