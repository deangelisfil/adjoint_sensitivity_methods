from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from function import Function
from parameters import *
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary


def bs_pde_adjoint(S0: float, sigma: float, r: float, option: Function, is_complex: bool = False):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    B = B_construction.evaluate()
    f = option.evaluate(S, is_complex)
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    qoi = np.dot(p, f)
    return qoi


def bs_pde_adjoint_f(S0: float, sigma: float, r: float):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    # f = option.evaluate(S, is_complex)
    # qoi = np.dot(p, f)
    return S, B, p