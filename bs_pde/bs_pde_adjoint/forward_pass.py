from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from function import Function
from parameters import grid
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary
from S_construction import S_construction
import numpy as np

def bs_pde_adjoint(S0: float, sigma: float, r: float, option: Function, is_complex: bool = False):
    s_construction = S_construction(S0, grid)
    S = s_construction.evaluate(is_complex)
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    B = B_construction.evaluate()
    f = option.evaluate(S, is_complex)
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    qoi = np.dot(p, f)
    return qoi


def bs_pde_adjoint_f(S0: float, sigma: float, r: float):
    s_construction = S_construction(S0, grid)
    S = s_construction.evaluate()
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    # f = option.evaluate(S, is_complex)
    # qoi = np.dot(p, f)
    return S, B, p