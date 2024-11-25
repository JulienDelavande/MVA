import numpy as np

def centering_step(Q, p, A, b, t, v0, eps):
    """
    Implements Newton's method to solve the centering step in the barrier method for QP.

    Parameters:
    Q   : numpy.ndarray
        Positive definite matrix in the quadratic term of the objective function (n x n).
    p   : numpy.ndarray
        Coefficient vector in the linear term of the objective function (n,).
    A   : numpy.ndarray
        Constraint matrix (m x n).
    b   : numpy.ndarray
        Constraint vector (m,).
    t   : float
        Barrier method parameter.
    v0  : numpy.ndarray
        Initial guess for the variables (n,).
    eps : float
        Target precision.

    Returns:
    v_seq : list of numpy.ndarray
        Sequence of variable iterates.
    """
    # Initialize variables
    v = v0.copy()
    v_seq = [v.copy()]
    m, n = A.shape

    # Backtracking line search parameters
    alpha = 0.01
    beta = 0.5

    # Precompute A v and b - A v
    def compute_phi_and_derivatives(v):
        # Compute barrier function value
        residual = b - A @ v
        if np.any(residual <= 0):
            # Return infinity if outside the domain
            return np.inf, None, None
        phi = t * (0.5 * v.T @ Q @ v + p.T @ v) - np.sum(np.log(residual))

        # Compute gradient
        grad = t * (Q @ v + p) + A.T @ (1 / residual)

        # Compute Hessian
        Hessian = t * Q + A.T @ np.diag(1 / (residual ** 2)) @ A

        return phi, grad, Hessian

    while True:
        phi, grad, Hessian = compute_phi_and_derivatives(v)

        # Compute Newton step and decrement
        try:
            # Solve Hessian * delta_v = -grad
            delta_v = -np.linalg.solve(Hessian, grad)
        except np.linalg.LinAlgError:
            # Hessian is singular
            print("Hessian is singular at iteration.")
            break

        lambda_squared = grad.T @ np.linalg.solve(Hessian, grad)

        # Check stopping criterion
        if lambda_squared / 2 <= eps:
            break

        # Backtracking line search
        step_size = 1.0
        while True:
            v_new = v + step_size * delta_v
            residual_new = b - A @ v_new
            if np.all(residual_new > 0):
                phi_new, _, _ = compute_phi_and_derivatives(v_new)
                if phi_new <= phi + alpha * step_size * grad.T @ delta_v:
                    break
            step_size *= beta

        # Update v
        v = v_new
        v_seq.append(v.copy())

    return v_seq
