"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""
import numpy as np

def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    X_train = np.zeros((n_train, max_train_card), dtype=int)
    y_train = np.zeros(n_train, dtype=int)
    M = np.random.randint(1, max_train_card + 1, n_train)

    for i in range(n_train):
        M_i = M[i]
        digits = np.random.randint(1, 11, M_i)
        X_train[i, -M_i:] = digits
        y_train[i] = digits.sum()
    ##############

    return X_train, y_train

def create_test_dataset():
    
    ############## Task 2
    n_samples_per_card = 10000
    cardinalities = np.arange(5, 101, 5)
    max_test_card = 100
    n_test = n_samples_per_card * len(cardinalities)

    X_test = np.zeros((n_test, max_test_card), dtype=int)
    y_test = np.zeros(n_test, dtype=int)

    idx = 0
    for c in cardinalities:
        digits = np.random.randint(1, 11, size=(n_samples_per_card, c))
        sums = digits.sum(axis=1)
        X_test[idx:idx + n_samples_per_card, -c:] = digits
        y_test[idx:idx + n_samples_per_card] = sums
        idx += n_samples_per_card
    ##############

    return X_test, y_test
