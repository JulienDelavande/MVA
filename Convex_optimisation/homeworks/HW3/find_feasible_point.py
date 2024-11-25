import numpy as np

def find_feasible_point(A, b):
    """
    Finds a feasible point v0 such that A v0 < b.

    Parameters:
    A : Constraint matrix (m x n).
    b : Constraint vector (m,).

    Returns:
    v0 : Feasible point (n,).
    """
    m, n = A.shape
    # Introduce slack variable s
    c = np.zeros(n + 1)
    c[-1] = 1  # Objective is to minimize s

    # Constraints: A v - s <= b
    A_aug = np.hstack([A, -np.ones((m, 1))])
    bounds = [(None, None)] * n + [(0, None)]  # s >= 0

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_aug, b_ub=b, bounds=bounds, method='highs')
    if res.success and res.x[-1] <= 1e-8:
        v0 = res.x[:-1]
        return v0
    else:
        raise ValueError("No feasible point found.")
