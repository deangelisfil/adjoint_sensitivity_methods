import numpy as np
from time_invariant_matrix_construction import *
from payoff import Payoff


def bs_pde_adjoint_auxiliary(B):
    p = np.zeros(2*J + 1)
    p[J] = 1
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
    return p

def bs_pde_adjoint(S0: float, sigma: float, r: float, option: Payoff, is_complex: bool = False):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B = B_construction(S, sigma, r)
    f = option.payoff(S, is_complex)
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
    B = B_construction(S, sigma, r)
    p_all_list = bs_pde_adjoint_auxiliary_f(B)
    return S, B, p_all_list