from parameters import *
import numpy as np
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.forward_pass import bs_pde_adjoint_auxiliary_f


def bs_pde_adjoint_auxiliary_b(B: np.ndarray, p_all_list: list, p_bar: float) -> tuple :
    d = 2 * J + 1
    B_transpose_bar = np.zeros((d, d))
    for n in reversed(range(N)) :
        B_transpose_bar = B_transpose_bar + np.outer(p_bar, p_all_list[n])
        p_bar = np.dot(B, p_bar)
    B_bar = B_transpose_bar.transpose()
    return B_bar


def bs_pde_adjoint_auxiliary_reverse(B: np.ndarray, p_bar: np.ndarray) :
    # forward pass
    p_all_list = bs_pde_adjoint_auxiliary_f(B)
    # reverse pass
    B_bar = bs_pde_adjoint_auxiliary_b(B, p_all_list, p_bar)
    return B_bar
