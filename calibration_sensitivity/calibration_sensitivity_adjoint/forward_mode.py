from B_construction.b_construction_time_invariant import *
from bs_pde.bs_pde_adjoint.forward_mode import bs_pde_adjoint_auxiliary_forward
from calibration_sensitivity.calibration_loss import Calibration_loss
import numpy as np
from parameters import *

def calibration_sensitivity_adjoint_forward(S0: float, sigma: float, r: float,
                                            diff_S0: float, diff_sigma: float, diff_r: float,
                                            option_list: list, loss: Calibration_loss):
    diff_S = diff_S0 * np.ones(2*J + 1)
    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_B = B_construction_time_invariant_forward(S, sigma, r, delta_t, delta_S, diff_S, diff_sigma, diff_r)
    B = B_construction_time_invariant_f(S, sigma, r, delta_t, delta_S)
    p, diff_p = bs_pde_adjoint_auxiliary_forward(B, diff_B)
    f = np.array(list(map(lambda x : x.evaluate(S), option_list)))
    diff_P_model = np.dot(f, diff_p)
    P_model = np.dot(f, p)
    diff_loss = np.dot(loss.diff_evaluate(P_model), diff_P_model)
    # loss = loss.evaluate(P_model)
    return diff_loss



