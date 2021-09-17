from parameters import *
import numpy as np


def bs_pde_adjoint_auxiliary(B):
    """Assumes that S_0 is in the middle of S"""
    p = np.zeros(2*J + 1)
    p[J] = 1
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
    return p


def bs_pde_adjoint_auxiliary_f(B):
    p = np.zeros(2*J + 1); p[J] = 1
    p_all_list = [p]
    B_transpose = B.transpose()
    for n in range(N):
        p = np.dot(B_transpose, p)
        p_all_list.append(p)
    return p_all_list
