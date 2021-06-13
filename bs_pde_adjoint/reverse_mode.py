from time_invariant_matrix_construction import *
from bs_pde_adjoint.forward_pass import bs_pde_adjoint_f


# reverse mode
def bs_pde_adjoint_b(S, sigma, r, B, p_all_list, qoi_bar) :
    d = 2 * J + 1
    B_transpose_bar = np.zeros((d, d))
    S_bar = p_all_list[-1] * diff_payoff(S) * qoi_bar  # first/second multiplication is elementwise/scalar
    p_bar = payoff(S) * qoi_bar
    for n in reversed(range(N)) :
        B_transpose_bar = B_transpose_bar + np.outer(p_bar, p_all_list[n])
        p_bar = np.dot(B, p_bar)
    B_bar = B_transpose_bar.transpose()
    S_bar, sigma_bar, r_bar = B_construction_reverse(B_bar, S_bar, S, sigma, r)
    S0_bar = sum(S_bar)
    return S0_bar, sigma_bar, r_bar


def bs_pde_adjoint_reverse(S0, sigma, r, qoi_bar=1) :
    # forward pass
    S, B, p_all_list = bs_pde_adjoint_f(S0, sigma, r)
    # backward pass
    S0_bar, sigma_bar, r_bar = bs_pde_adjoint_b(S, sigma, r, B, p_all_list, qoi_bar)
    return S0_bar, sigma_bar, r_bar
