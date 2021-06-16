from calibration_sensitivity.calibration_loss import Calibration_loss
from bs_pde.bs_pde_standard.forward_mode import bs_pde_standard_forward
import numpy as np


def calibration_sensitivity_standard_forward(S0: float, sigma: float, r: float,
                                             diff_S0: float, diff_sigma: float, diff_r: float,
                                             option_list: list, loss: Calibration_loss,
                                             american=False) :
    K = len(option_list)
    diff_P_model = np.zeros(K); P_model = np.zeros(K)
    for k in range(K):
        P_model[k], diff_P_model[k] = bs_pde_standard_forward(S0, sigma, r, diff_S0, diff_sigma, diff_r, option_list[k], american)
    diff_loss = np.dot(loss.diff_evaluate(P_model), diff_P_model)
    loss = loss.evaluate(P_model)
    return loss, diff_loss

