"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    ##################
    walk = [node]
    
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors)
        walk.append(next_node)
    ##################
    
    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    for node in G.nodes():
        for _ in range(num_walks):
            walk = random_walk(G, node, walk_length)
            walks.append(walk)
    
    np.random.shuffle(walks)
    ##################

    return walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
