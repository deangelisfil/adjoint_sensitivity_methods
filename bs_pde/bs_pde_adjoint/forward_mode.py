from B_construction.b_construction_time_invariant import *
from function import Function
from parameters import *

def bs_pde_adjoint_auxiliary_forward(B, diff_B):
    diff_p = np.zeros(2 * J + 1)
    p = np.zeros(2*J + 1); p[J] = 1
    diff_B_transpose = diff_B.transpose()
    B_transpose = B.transpose()
    for n in range(N):
        diff_p = np.dot(B_transpose, diff_p) + np.dot(diff_B_transpose, p)
        p = np.dot(B_transpose, p)
    return p, diff_p



def bs_pde_adjoint_forward(S0: float, sigma: float, r: float,
                           diff_S0: float, diff_sigma: float, diff_r: float,
                           option: Function):
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_B = B_construction_time_invariant_forward(S, sigma, r, delta_t, delta_S, diff_S, diff_sigma, diff_r)
    B = B_construction_time_invariant_f(S, sigma, r, delta_t, delta_S)
    p, diff_p = bs_pde_adjoint_auxiliary_forward(B, diff_B)
    diff_f = option.diff_evaluate(S) * diff_S
    f = option.evaluate(S)
    diff_qoi = np.dot(p, diff_f) + np.dot(diff_p, f)
    # qoi = np.dot(p, f)
    return diff_qoi