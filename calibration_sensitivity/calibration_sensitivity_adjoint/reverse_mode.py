from calibration_sensitivity.calibration_sensitivity_adjoint.forward_pass import calibration_sensitivity_adjoint_f
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary
from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from calibration_sensitivity.calibration_loss import Calibration_loss
from parameters import *


def calibration_sensitivity_adjoint_b(S: np.ndarray,
                                      sigma: float,
                                      r: float,
                                      B: np.ndarray,
                                      f: np.ndarray,
                                      P_model: np.ndarray,
                                      loss: Calibration_loss,
                                      loss_bar: float) :
    P_model_bar = loss.diff_evaluate(P_model) * loss_bar
    p_bar = np.dot(f.transpose(), P_model_bar)
    # No f_bar = np.outer(P_model_bar, p_all_list[-1]) because f does not depend on theta = {S0, sigma, r}
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    B_bar = bs_pde_adjoint_auxiliary.reverse(p_bar)
    # B_bar = bs_pde_adjoint_auxiliary_b(B, p_all_list, p_bar)
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    S_bar, sigma_bar, r_bar = B_construction.reverse(B_bar)
    # S0_bar = sum(S_bar) # ignore S0
    return sigma_bar, r_bar


def calibration_sensitivity_adjoint_reverse(S: np.ndarray,
                                            sigma: float,
                                            r: float,
                                            option_list: list,
                                            loss: Calibration_loss,
                                            loss_bar: float = 1) :
    # forward pass
    B, f, P_model = calibration_sensitivity_adjoint_f(S, sigma, r, option_list)
    # backward pass
    sigma_bar, r_bar = calibration_sensitivity_adjoint_b(S, sigma, r, B, f, P_model, loss, loss_bar)
    return sigma_bar, r_bar
