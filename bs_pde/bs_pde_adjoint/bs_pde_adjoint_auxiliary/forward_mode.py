from parameters import *
import numpy as np

def bs_pde_adjoint_auxiliary_forward(B, diff_B):
    diff_p = np.zeros(2 * J + 1)
    p = np.zeros(2*J + 1); p[J] = 1
    diff_B_transpose = diff_B.transpose()
    B_transpose = B.transpose()
    for n in range(N):
        diff_p = B_transpose.dot(diff_p) + diff_B_transpose.dot(p)
        p = B_transpose.dot(p)
    return diff_p
