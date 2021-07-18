from B_construction.b_construction_time_invariant import *
from function import Function
from parameters import *

def bs_pde_adjoint_auxiliary(B):
    p = np.zeros(2*J + 1)
    p[J] = 1
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
    return p

def bs_pde_adjoint(S0: float, sigma: float, r: float, option: Function, is_complex: bool = False):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B = B_construction_time_invariant_f(S, sigma, r, delta_t, delta_S)
    f = option.evaluate(S, is_complex)
    p = bs_pde_adjoint_auxiliary(B)
    qoi = np.dot(p, f)
    return qoi


def bs_pde_adjoint_auxiliary_f(B):
    p = np.zeros(2*J + 1); p[J] = 1
    p_all_list = [p]
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
        p_all_list.append(p)
    return p_all_list


def bs_pde_adjoint_f(S0: float, sigma: float, r: float):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B = B_construction_time_invariant_f(S, sigma, r, delta_t, delta_S)
    p_all_list = bs_pde_adjoint_auxiliary_f(B)
    # f = option.evaluate(S, is_complex)
    # qoi = np.dot(p, f)
    return S, B, p_all_list