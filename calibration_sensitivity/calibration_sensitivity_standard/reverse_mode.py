from calibration_sensitivity.calibration_sensitivity_standard.forward_pass import calibration_sensitivity_standard_f
from bs_pde.bs_pde_standard.reverse_mode import bs_pde_standard_b
from calibration_sensitivity.calibration_loss import Calibration_loss
import numpy as np


def calibration_sensitivity_standard_b(S: float, sigma: float, r: float, option_list: list, loss: Calibration_loss,
                                       american: bool, B: np.ndarray, u_all_payoff_list: list, u_hat_all_payoff_list: list,
                                       P_model: np.ndarray, loss_bar: float=1) -> tuple:
    S0_bar = 0; sigma_bar = 0; r_bar = 0
    K = len(option_list)
    P_model_bar = loss.diff_evaluate(P_model) * loss_bar

    for k in range(K):
        S0_bar_delta, sigma_bar_delta, r_bar_delta = bs_pde_standard_b(S, sigma, r, option_list[k], B,
                                                                       u_all_payoff_list[k], u_hat_all_payoff_list[k],
                                                                       P_model_bar[k], american)
        S0_bar += S0_bar_delta
        sigma_bar += sigma_bar_delta
        r_bar += r_bar_delta
    # ignore S0_bar
    return sigma_bar, r_bar


def calibration_sensitivity_standard_reverse(S0: float, sigma: float, r: float, option_list: list, loss: Calibration_loss,
                                             american: bool, loss_bar: float = 1):
    # forward pass
    P_model, S, B, u_all_payoff_list, u_hat_all_payoff_lis = calibration_sensitivity_standard_f(S0, sigma, r, option_list, loss,
                                                                                                american)
    # backward pass
    sigma_bar, r_bar = calibration_sensitivity_standard_b(S, sigma, r, option_list, loss, american, B,
                                                                  u_all_payoff_list, u_hat_all_payoff_lis, P_model, loss_bar)
    return sigma_bar, r_bar

