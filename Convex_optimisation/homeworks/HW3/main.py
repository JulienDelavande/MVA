import numpy as np
import matplotlib.pyplot as plt
from barr_method import barr_method
from find_feasible_point import find_feasible_point

# Generate random data
np.random.seed(42)
n = 100
d = 20
X = np.random.randn(n, d)
true_w = np.zeros(d)
sparse_indices = np.random.choice(range(d), size=5, replace=False)
true_w[sparse_indices] = np.random.randn(5)
y = X @ true_w + 0.1 * np.random.randn(n)
lambda_reg = 10

# Define QP parameters
Q = np.eye(n)
p = y
A = np.vstack([X.T, -X.T])
b = lambda_reg * np.ones(2 * d)

# Find feasible point
v0 = find_feasible_point(A, b)
eps = 1e-6
mu_values = [2, 15, 50, 100]
results = {}

for mu in mu_values:
    print(f"Running barrier method with mu = {mu}")
    v_seq, gap_seq = barr_method(Q, p, A, b, v0, eps, mu=mu)
    results[mu] = {
        'v_seq': v_seq,
        'gap_seq': gap_seq,
        'final_v': v_seq[-1]
    }

def dual_objective(v):
    return 0.5 * np.linalg.norm(v)**2 + y.T @ v

# Find f*
f_values = []
for mu in mu_values:
    final_v = results[mu]['final_v']
    f_val = dual_objective(final_v)
    f_values.append(f_val)
f_star = min(f_values)
print(f"Best objective value found (f*): {f_star}")

# Compute f(v_t) - f*
for mu in mu_values:
    v_seq = results[mu]['v_seq']
    f_gap = []
    for v in v_seq:
        f_val = dual_objective(v)
        f_gap.append(f_val - f_star)
    results[mu]['f_gap'] = f_gap

# Plot convergence
plt.figure(figsize=(12, 6))
for mu in mu_values:
    f_gap = results[mu]['f_gap']
    plt.semilogy(f_gap, label=f'μ = {mu}')
plt.xlabel('Iteration')
plt.ylabel('Objective Gap f(v_t) - f*')
plt.title('Convergence of Barrier Method for Different μ Values')
plt.legend()
plt.grid(True)
plt.show()

# Compute w and plot
plt.figure(figsize=(12, 6))
for mu in mu_values:
    final_v = results[mu]['final_v']
    w = -X.T @ final_v
    results[mu]['w'] = w
    plt.plot(w, label=f'μ = {mu}')
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.title('Weights w Obtained for Different μ Values')
plt.legend()
plt.grid(True)
plt.show()
