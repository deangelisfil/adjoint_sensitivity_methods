import numpy as np
from time_invariant_matrix_construction import *

# forward pass
def bs_pde_adjoint_auxiliary(f, B):
    p = np.zeros(2*J + 1)
    p[J] = 1
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
    qoi = np.dot(p, f)
    return qoi

def bs_pde_adjoint(S0, sigma, r, is_complex = False):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B = B_construction(S, sigma, r)
    f = payoff(S, is_complex)
    return bs_pde_adjoint_auxiliary(f, B)


def bs_pde_adjoint_auxiliary_f(f, B):
    p = np.zeros(2*J + 1); p[J] = 1
    p_all_list = [p]
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
        p_all_list.append(p)
    qoi = np.dot(p, f)
    return p_all_list


def bs_pde_adjoint_f(S0, sigma, r):
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    B = B_construction(S, sigma, r)
    f = payoff(S)
    p_all_list = bs_pde_adjoint_auxiliary_f(f, B)
    return S, B, p_all_list