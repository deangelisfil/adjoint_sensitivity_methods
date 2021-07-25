from calibration_sensitivity.calibration_loss import Calibration_loss
from bs_pde.bs_pde_standard.forward_mode import bs_pde_standard_forward
import numpy as np
from bs_pde.bs_pde_standard.bs_pde_standard import Bs_pde_standard


def calibration_sensitivity_standard_forward(S0: float, sigma: float, r: float,
                                             diff_sigma: float, diff_r: float,
                                             option_list: list, loss: Calibration_loss,
                                             american=False) :
    diff_S0 = 0 # view S0 as fixed
    K = len(option_list)
    diff_P_model = np.zeros(K); P_model = np.zeros(K)
    for k in range(K):
        bs_pde_standard = Bs_pde_standard(S0, sigma, r, option_list[k], american)
        P_model[k] = bs_pde_standard.evaluate()
        diff_P_model[k] = bs_pde_standard.forward([diff_S0, diff_sigma, diff_r])
    diff_loss = np.dot(loss.diff_evaluate(P_model), diff_P_model)
    # loss = loss.evaluate(P_model)
    return diff_loss

