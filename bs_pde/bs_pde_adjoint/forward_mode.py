from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from function import Function
from parameters import grid
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary
from S_construction import S_construction
import numpy as np


def bs_pde_adjoint_forward(S0: float, sigma: float, r: float,
                           diff_S0: float, diff_sigma: float, diff_r: float,
                           option: Function):
    s_construction = S_construction(S0, grid)
    diff_S = s_construction.forward([diff_S0])
    S = s_construction.evaluate()
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    diff_B = B_construction.forward([diff_S, diff_sigma, diff_r])
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    diff_p = bs_pde_adjoint_auxiliary.forward([diff_B])
    p = bs_pde_adjoint_auxiliary.evaluate()
    diff_f = option.diff_evaluate(S) * diff_S
    f = option.evaluate(S)
    diff_qoi = np.dot(p, diff_f) + np.dot(diff_p, f)
    # qoi = np.dot(p, f)
    return diff_qoi