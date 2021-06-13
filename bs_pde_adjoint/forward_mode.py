from time_invariant_matrix_construction import *

# forward mode
def bs_pde_adjoint_forward(S0, sigma, r, diff_S0, diff_sigma, diff_r):
    diff_p = np.zeros(2*J + 1)
    p = np.zeros(2*J + 1); p[J] = 1
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_B_transpose = diff_B_construction(S, sigma, r, diff_S, diff_sigma, diff_r).transpose()
    B_transpose = B_construction(S, sigma, r).transpose()
    for n in range(N):
        diff_p = np.dot(B_transpose, diff_p) + np.dot(diff_B_transpose, p)
        p = np.dot(B_transpose, p)
    diff_f = diff_payoff(S)*diff_S
    f = payoff(S)
    diff_qoi = np.dot(p, diff_f) + np.dot(diff_p, f)
    qoi = np.dot(p, f)
    return qoi, diff_qoi