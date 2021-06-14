from time_invariant_matrix_construction import *
from bs_pde_adjoint.forward_pass import bs_pde_adjoint_auxiliary, bs_pde_adjoint_auxiliary_f


def calibration_sensitivity(S0: float , sigma: float, r: float, option_list: list, loss_fn: callable) :
    S = np.array([S0 + j * delta_S for j in range(-J, J + 1)])
    B = B_construction(S, sigma, r)
    p = bs_pde_adjoint_auxiliary(B)
    f = np.array(list(map(lambda x : x.payoff(S), option_list)))
    P_model = np.dot(f, p)
    loss = loss_fn(P_model)
    return loss

def calibration_sensitivity_f(S0: float , sigma: float, r: float, option_list: list) :
    S = np.array([S0 + j * delta_S for j in range(-J, J + 1)])
    B = B_construction(S, sigma, r)
    p_all_list = bs_pde_adjoint_auxiliary_f(B)
    f = np.array(list(map(lambda x : x.payoff(S), option_list)))
    P_model = np.dot(f, p_all_list[-1])
    return S, B, p_all_list, f, P_model

