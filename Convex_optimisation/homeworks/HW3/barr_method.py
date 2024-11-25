from centering_step import centering_step

def barr_method(Q, p, A, b, v0, eps, mu=10, max_iter=50):
    """
    Implements the barrier method to solve a quadratic program (QP).
    
    Parameters:
    Q, p, A, b : QP parameters.
    v0         : Initial feasible point.
    eps        : Target precision for the duality gap.
    mu         : Multiplicative factor to increase t.
    max_iter   : Maximum number of outer iterations.

    Returns:
    v_seq      : Sequence of variable iterates.
    gap_seq    : Sequence of duality gaps.
    """
    t = 1.0       # Initial barrier parameter
    m = len(b)    # Number of inequality constraints
    v = v0.copy()
    v_seq = [v.copy()]
    gap_seq = []
    
    for _ in range(max_iter):
        # Centering step
        v_seq_center = centering_step(Q, p, A, b, t, v, eps)
        v = v_seq_center[-1]
        v_seq.extend(v_seq_center[1:])
        
        # Duality gap
        duality_gap = m / t
        gap_seq.append(duality_gap)
        
        if duality_gap <= eps:
            break
        else:
            t *= mu
    
    return v_seq, gap_seq
