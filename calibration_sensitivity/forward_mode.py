from time_invariant_matrix_construction import *
from bs_pde_adjoint.forward_pass import bs_pde_adjoint_auxiliary, bs_pde_adjoint_auxiliary_f
from bs_pde_adjoint.forward_mode import bs_pde_adjoint_auxiliary_forward
import numpy as np

def calibration_sensitivity_forward(S0: float , sigma: float, r: float,
                                    diff_S0: float, diff_sigma: float, diff_r: float,
                                    option_list: list, loss_fn: callable, diff_loss_fn: callable):
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_B = diff_B_construction(S, sigma, r, diff_S, diff_sigma, diff_r)
    B = B_construction(S, sigma, r)
    p, diff_p = bs_pde_adjoint_auxiliary_forward(B, diff_B)
    f = np.array(list(map(lambda x : x.payoff(S), option_list)))
    diff_P_model = np.dot(f, diff_p)
    P_model = np.dot(f, p)
    diff_loss = np.dot(diff_loss_fn(P_model), diff_P_model)
    loss = loss_fn(P_model)
    return loss, diff_loss



