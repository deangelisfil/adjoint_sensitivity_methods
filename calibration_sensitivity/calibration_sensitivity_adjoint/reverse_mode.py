from calibration_sensitivity.calibration_sensitivity_adjoint.forward_pass import calibration_sensitivity_adjoint_f
from bs_pde.bs_pde_adjoint.reverse_mode import bs_pde_adjoint_auxiliary_b
from time_invariant_matrix_construction import *
from calibration_sensitivity.calibration_loss import Calibration_loss
import numpy as np

def calibration_sensitivity_adjoint_b(S: float, sigma: float, r: float, B: np.ndarray, p_all_list: list,
                                      f: np.ndarray, P_model: np.ndarray, loss: Calibration_loss, loss_bar: float):
    P_model_bar = loss.diff_evaluate(P_model) * loss_bar
    p_bar = np.dot(f.transpose(), P_model_bar)
    # No f_bar = np.outer(P_model_bar, p_all_list[-1]) because f does not depend on theta = {S0, sigma, r}
    B_bar = bs_pde_adjoint_auxiliary_b(B, p_all_list, p_bar)
    S_bar = np.zeros(S.shape)
    S_bar, sigma_bar, r_bar = B_construction_reverse(B_bar, S_bar, S, sigma, r)
    S0_bar = sum(S_bar)
    return S0_bar, sigma_bar, r_bar

def calibration_sensitivity_adjoint_reverse(S0: float, sigma: float, r: float, option_list: list, loss: Calibration_loss,
                                            loss_bar: float = 1):
    # forward pass
    S, B, p_all_list, f, P_model = calibration_sensitivity_adjoint_f(S0, sigma, r, option_list)
    # backward pass
    S0_bar, sigma_bar, r_bar = calibration_sensitivity_adjoint_b(S, sigma, r, B, p_all_list, f, P_model, loss, loss_bar)
    return S0_bar, sigma_bar, r_bar

