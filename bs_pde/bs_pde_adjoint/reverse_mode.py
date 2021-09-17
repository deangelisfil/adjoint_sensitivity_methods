from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from bs_pde.bs_pde_adjoint.forward_pass import bs_pde_adjoint_f
from function import Function
from parameters import *
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary


def bs_pde_adjoint_b(S: float, sigma: float, r: float, option: Function, B: np.ndarray, p: list,
                     qoi_bar: float = 1) :
    S_bar = p * option.diff_evaluate(S) * qoi_bar  # first/second multiplication is elementwise/scalar
    p_bar = option.evaluate(S) * qoi_bar
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    B_bar = bs_pde_adjoint_auxiliary.reverse(p_bar)
    # B_bar = bs_pde_adjoint_auxiliary_b(B, p_all_list, p_bar)
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    S_bar_, sigma_bar, r_bar = B_construction.reverse(B_bar, S_bar = S_bar)
    S0_bar = sum(S_bar)
    return S0_bar, sigma_bar, r_bar


def bs_pde_adjoint_reverse(S0: float, sigma: float, r: float, option: Function, qoi_bar: float = 1) :
    # forward pass
    S, B, p = bs_pde_adjoint_f(S0, sigma, r)
    # backward pass
    S0_bar, sigma_bar, r_bar = bs_pde_adjoint_b(S, sigma, r, option, B, p, qoi_bar)
    return S0_bar, sigma_bar, r_bar
