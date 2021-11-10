from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from calibration_sensitivity.calibration_loss import Calibration_loss
from parameters import *
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.bs_pde_adjoint_auxiliary import Bs_pde_adjoint_auxiliary

def calibration_sensitivity_adjoint_forward(S: np.ndarray, sigma: float, r: float, diff_sigma: float, diff_r: float,
                                            option_list: list, loss: Calibration_loss):
    diff_S = np.zeros(2*J + 1) # S is seen as fixed
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    diff_B = B_construction.forward([diff_S, diff_sigma, diff_r])
    B = B_construction.evaluate()
    bs_pde_adjoint_auxiliary = Bs_pde_adjoint_auxiliary(B)
    diff_p = bs_pde_adjoint_auxiliary.forward([diff_B])
    p = bs_pde_adjoint_auxiliary.evaluate()
    f = np.array(list(map(lambda x : x.evaluate(S), option_list)))
    diff_P_model = np.dot(f, diff_p)
    P_model = np.dot(f, p)
    diff_loss = np.dot(loss.diff_evaluate(P_model), diff_P_model)
    # loss = loss.evaluate(P_model)
    return diff_loss



