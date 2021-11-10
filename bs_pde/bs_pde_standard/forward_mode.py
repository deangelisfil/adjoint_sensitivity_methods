from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from auxiliary_functions import maximum
from function import Function
from bs_pde.bs_pde_standard.bs_pde_standard_auxiliary import Bs_pde_standard_auxiliary
from S_construction import S_construction
from Grid import Grid

# def bs_pde_standard_auxiliary_pde_forward(f, B, diff_f, diff_B):
#     diff_b_all_list = []
#     diff_u = diff_f
#     u = np.copy(f)
#     for n in reversed(range(N)):
#         diff_b = np.dot(diff_B, u)
#         diff_u = np.dot(B, diff_u) + diff_b
#         u = np.dot(B, u)
#         diff_b_all_list.append(diff_b)
#     diff_qoi = diff_u[J]
#     qoi = u[J]
#     return qoi, diff_qoi, list(reversed(diff_b_all_list)) #reverse diff_b list to be forward in time


def bs_pde_standard_forward(S0: float,
                            sigma: float,
                            r: float,
                            diff_S0: float,
                            diff_sigma: float,
                            diff_r: float,
                            option: Function,
                            grid: Grid,
                            american: bool):
    s_construction = S_construction(S0, grid)
    diff_S = s_construction.forward([diff_S0])
    S = s_construction.evaluate()
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    diff_B = B_construction.forward([diff_S, diff_sigma, diff_r])
    B = B_construction.evaluate()
    diff_f = option.diff_evaluate(S) * diff_S
    f = option.evaluate(S)
    bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f,
                                                          grid_local = grid,
                                                          american = american)
    diff_qoi = bs_pde_standard_auxiliary.forward([diff_B, diff_f])
    # qoi = bs_pde_standard_auxiliary.evaluate()
    return diff_qoi