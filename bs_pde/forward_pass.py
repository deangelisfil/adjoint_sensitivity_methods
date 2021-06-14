import numpy as np
from time_invariant_matrix_construction import *
from auxiliary_functions import maximum
from payoff import Payoff


def bs_pde_auxiliary(f, B, american=False, is_complex=False) :
    u = np.copy(f)
    for n in reversed(range(N)) :
        u = np.dot(B, u)
        if american :
            u = maximum(u, f, is_complex)
    qoi = u[J]
    return qoi


def bs_pde(S0 : float,
           sigma : float,
           r : float,
           option : Payoff,
           american : bool = False,
           is_complex: bool = False) -> float:
    S = np.array([S0 + j * delta_S for j in range(-J, J + 1)])
    B = B_construction(S, sigma, r)
    f = option.payoff(S, is_complex)
    return bs_pde_auxiliary(f, B, american, is_complex)


def bs_pde_auxiliary_f(f, B, american=False) :
    u = np.copy(f)
    u_all_list = [u]
    u_hat_all_list = []
    for n in reversed(range(N)):
        u = np.dot(B, u)
        if american :
            u_hat_all_list.append(u)
            u = maximum(u, f, is_complex=False)
        u_all_list.append(u)
    # qoi = u[J]
    return list(reversed(u_all_list)), list(reversed(u_hat_all_list))


def bs_pde_f(S0: float, sigma: float, r: float, option: Payoff, american: bool = False) -> tuple:
    S = np.array([S0 + j * delta_S for j in range(-J, J + 1)])
    B = B_construction(S, sigma, r)
    f = option.payoff(S)
    u_all_list, u_hat_all_list = bs_pde_auxiliary_f(f, B, american)
    return S, B, u_all_list, u_hat_all_list
