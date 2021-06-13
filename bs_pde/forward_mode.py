import numpy as np
from time_invariant_matrix_construction import *
from auxiliary_functions import maximum

# forward mode
def bs_pde_auxiliary_forward(f, B, diff_f, diff_B):
    diff_b_all_list = []
    diff_u = diff_f
    u = np.copy(f)
    for n in reversed(range(N)):
        diff_b = np.dot(diff_B, u)
        diff_u = np.dot(B, diff_u) + diff_b
        u = np.dot(B, u)
        diff_b_all_list.append(diff_b)
    diff_qoi = diff_u[J]
    qoi = u[J]
    return qoi, diff_qoi, list(reversed(diff_b_all_list)) #reverse diff_b list to be forward in time


def bs_pde_forward(S0, sigma, r, diff_S0, diff_sigma, diff_r, american = False):
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_u = diff_payoff(S) * diff_S
    u = payoff(S)
    diff_B = diff_B_construction(S, sigma, r, diff_S, diff_sigma, diff_r)
    B = B_construction(S, sigma, r)
    for n in reversed(range(N)):
        diff_u = np.dot(B, diff_u) + np.dot(diff_B, u)
        u = np.dot(B, u)
        if american:
#           holding  = heaviside_close(u-payoff(S), 1)
            holding  = np.heaviside(u-payoff(S), 1)
            # print(holding)
            diff_u = diff_u * holding + diff_payoff(S) * diff_S * (1-holding)
            u = maximum(u, payoff(S), is_complex=False)
    diff_qoi = diff_u[J]
    qoi = u[J]
    return qoi, diff_qoi