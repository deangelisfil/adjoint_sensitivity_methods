from calibration_sensitivity.calibration_loss import Calibration_loss
from bs_pde.bs_pde_standard.forward_pass import bs_pde_standard, bs_pde_standard_f
import numpy as np

def calibration_sensitivity_standard(S0: float, sigma: float, r: float, option_list: list, loss: Calibration_loss,
                                     american, is_complex=False):
    K = len(option_list)
    P_model = np.zeros(K)
    for k in range(K):
        P_model[k] = bs_pde_standard(S0, sigma, r, option_list[k], american, is_complex) # TO DO: simplify to not repeat S and B construction, same applies for bs_pde_f and forward mode
    loss = loss.evaluate(P_model)
    return loss

def calibration_sensitivity_standard_f(S0: float, sigma: float, r: float, option_list: list, loss: Calibration_loss,
                                       american):
    K = len(option_list)
    P_model = np.zeros(K)
    u_all_payoff_list = []
    u_hat_all_payoff_list = []
    for k in range(K):
        P_model[k], S, B, u_all_list, u_hat_all_list = bs_pde_standard_f(S0, sigma, r, option_list[k], american)
        u_all_payoff_list.append(u_all_list)
        u_hat_all_payoff_list.append(u_hat_all_list)
    # loss = loss.evaluate(P_model)
    return P_model, S, B, u_all_payoff_list, u_hat_all_payoff_list