from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from function import Function
from parameters import *
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary

def bs_pde_adjoint_forward(S0: float, sigma: float, r: float,
                           diff_S0: float, diff_sigma: float, diff_r: float,
                           option: Function):
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
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