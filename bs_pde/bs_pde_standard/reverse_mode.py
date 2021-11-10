from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from bs_pde.bs_pde_standard.forward_pass import bs_pde_standard_f
from bs_pde.bs_pde_standard.bs_pde_standard_auxiliary import Bs_pde_standard_auxiliary
from function  import Function
from S_construction import S_construction
import numpy as np
from Grid import Grid

# # reverse mode
# def bs_pde_standard_auxiliary_b(B, qoi_bar=1) :
#     d = 2 * J + 1
#     u_bar = np.zeros(d)
#     u_bar[J] = qoi_bar
#     b_bar_all_list = []
#     B_transpose = B.transpose()
#     for n in range(N) :
#         b_bar = u_bar
#         u_bar = np.dot(B_transpose, u_bar)
#         b_bar_all_list.append(b_bar)
#     f_bar = u_bar
#     return f_bar, b_bar_all_list
#
#
# def bs_pde_standard_auxiliary_reverse(B, qoi_bar=1) :
#     # forward pass
#     # u_all_list = bs_pde_auxiliary_f(f, B)
#     # backward pass
#     f_bar, b_bar_all_list = bs_pde_standard_auxiliary_b(B, qoi_bar)
#     return f_bar, b_bar_all_list


def bs_pde_standard_b(S0: float,
                      sigma: float,
                      r: float,
                      option: Function,
                      B: np.ndarray,
                      f: np.ndarray,
                      grid: Grid,
                      american: bool,
                      qoi_bar: float) -> tuple:
    bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f,
                                                          grid_local = grid,
                                                          american = american)
    B_bar, f_bar = bs_pde_standard_auxiliary.reverse(qoi_bar)
    s_construction = S_construction(S0, grid)
    S = s_construction.evaluate()
    S_bar = option.diff_evaluate(S) * f_bar
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    S_bar, sigma_bar, r_bar = B_construction.reverse(B_bar, S_bar = S_bar)
    # S_bar = S_bar + option.diff_evaluate(S) * u_bar  # elementwise multiplication
    S0_bar = s_construction.reverse(S_bar)
    return S0_bar, sigma_bar, r_bar


def bs_pde_standard_reverse(S0, sigma, r, option, grid, american, qoi_bar) :
    # forward pass
    qoi, B, f = bs_pde_standard_f(S0, sigma, r, option, grid, american)
    # backward pass
    S0_bar, sigma_bar, r_bar = bs_pde_standard_b(S0, sigma, r, option, B, f, grid, american, qoi_bar)
    return S0_bar, sigma_bar, r_bar