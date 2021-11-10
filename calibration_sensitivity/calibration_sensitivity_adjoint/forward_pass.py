from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from calibration_sensitivity.calibration_loss import Calibration_loss
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary
from parameters import *

def calibration_sensitivity_adjoint(S: np.ndarray, sigma: float, r: float, option_list: list, loss: Calibration_loss) :
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    f = np.array(list(map(lambda x : x.evaluate(S), option_list)))
    P_model = np.dot(f, p)
    loss = loss.evaluate(P_model)
    return loss

def calibration_sensitivity_adjoint_f(S: np.ndarray, sigma: float, r: float, option_list: list) :
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    p = bs_pde_adjoint_auxiliary.evaluate()
    f = np.array(list(map(lambda x : x.evaluate(S), option_list)))
    P_model = np.dot(f, p)
    return B, f, P_model

