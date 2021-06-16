from time_invariant_matrix_construction import *
from bs_pde.bs_pde_adjoint.forward_pass import bs_pde_adjoint_f
from function import Function


def bs_pde_adjoint_auxiliary_b(B: np.ndarray, p_all_list: list, p_bar: float) -> tuple:
    d = 2 * J + 1
    B_transpose_bar = np.zeros((d, d))
    for n in reversed(range(N)) :
        B_transpose_bar = B_transpose_bar + np.outer(p_bar, p_all_list[n])
        p_bar = np.dot(B, p_bar)
    B_bar = B_transpose_bar.transpose()
    return B_bar


def bs_pde_adjoint_b(S: float, sigma: float, r: float, option: Function, B: np.ndarray, p_all_list: list, qoi_bar: float = 1) :
    S_bar = p_all_list[-1] * option.diff_evaluate(S) * qoi_bar  # first/second multiplication is elementwise/scalar
    p_bar = option.evaluate(S) * qoi_bar
    B_bar = bs_pde_adjoint_auxiliary_b(B, p_all_list, p_bar)
    S_bar, sigma_bar, r_bar = B_construction_reverse(B_bar, S_bar, S, sigma, r)
    S0_bar = sum(S_bar)
    return S0_bar, sigma_bar, r_bar


def bs_pde_adjoint_reverse(S0: float, sigma: float, r: float, option: Function, qoi_bar: float = 1) :
    # forward pass
    S, B, p_all_list = bs_pde_adjoint_f(S0, sigma, r)
    # backward pass
    S0_bar, sigma_bar, r_bar = bs_pde_adjoint_b(S, sigma, r, option, B, p_all_list, qoi_bar)
    return S0_bar, sigma_bar, r_bar
