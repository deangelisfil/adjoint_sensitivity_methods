from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from function import Function
from bs_pde.bs_pde_standard.bs_pde_standard_auxiliary import Bs_pde_standard_auxiliary
from S_construction import S_construction
import numpy as np
from grid import Grid

def bs_pde_standard(S0 : float,
                    sigma : float,
                    r : float,
                    option : Function,
                    grid: Grid,
                    american : bool,
                    is_complex: bool) -> float:
    s_construction = S_construction(S0, grid)
    S = s_construction.evaluate(is_complex)
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    B = B_construction.evaluate()
    f = option.evaluate(S, is_complex)
    bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f,
                                                          grid_local = grid,
                                                          american = american)
    qoi = bs_pde_standard_auxiliary.evaluate(is_complex)
    return qoi


def bs_pde_standard_f(S0: float,
                      sigma: float,
                      r: float,
                      option: Function,
                      grid: Grid,
                      american: bool) -> tuple:
    s_construction = S_construction(S0, grid)
    S = s_construction.evaluate()
    B_construction = B_construction_time_invariant(S, sigma, r, grid.delta_t, grid.delta_S)
    B = B_construction.evaluate()
    f = option.evaluate(S)
    bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f,
                                                          grid_local = grid,
                                                          american = american)
    qoi = bs_pde_standard_auxiliary.evaluate()
    return qoi, B, f
