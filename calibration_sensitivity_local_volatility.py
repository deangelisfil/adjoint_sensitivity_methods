from calibration_sensitivity.calibration_loss import Calibration_loss
from auxiliary_functions import check_forward_reverse_mode_identity
import numpy as np
from black_box import Black_box
from splines.Cubic_splines_construction import Cubic_splines_construction
from splines.Cubic_splines_evaluation import Cubic_splines_evaluation
from calibration_sensitivity.calibration_sensitivity_adjoint.calibration_sensitivity_adjoint import \
    Calibration_sensitivity_adjoint
from calibration_sensitivity.calibration_sensitivity_standard.calibration_sensitivity_standard import \
    Calibration_sensitivity_standard
from calibration_sensitivity.calibration_sensitivity_abstract import Calibration_sensitivity_abstract


class Calibration_sensitivity_local_volatility(Black_box) :
    def __init__(self, S_p: np.ndarray, sigma_p: np.ndarray, S: np.ndarray, r: float, option_list: list,
                 loss: Calibration_loss, is_adjoint : bool = True, american : bool = False) :
        self.S_p = S_p
        self.sigma_p = sigma_p
        self.S = S
        self.r = r
        self.option_list = option_list
        self.loss = loss
        self.is_adjoint = is_adjoint
        self.american = american

    def __repr__(self) :
        return "Calibration sensitivity adjoint local volatility: S_p=" + str(self.S_p)

    def copy(self) :
        return Calibration_sensitivity_local_volatility(self.S_p, self.sigma_p, self.S, self.r,
                                                        self.option_list, self.loss)

    def get_u(self) -> list :
        return [self.sigma_p]

    def set_u(self, u: list) :
        assert len(u) == 1
        self.sigma_p = np.array(u[0])

    def get_calibration_sensitivity(self, sigma) -> Calibration_sensitivity_abstract:
        if self.is_adjoint:
            calibration_sensitivity = Calibration_sensitivity_adjoint(self.S, sigma, self.r, self.option_list,
                                                                      self.loss)
        else:
            calibration_sensitivity = Calibration_sensitivity_standard(self.S, sigma, self.r, self.option_list,
                                                                       self.loss, self.american)
        return calibration_sensitivity

    def evaluate(self, is_complex=False) :
        cubic_splines_construction = Cubic_splines_construction(self.S_p, self.sigma_p)
        cubic_splines_evaluation = Cubic_splines_evaluation(self.S, cubic_splines_construction)
        sigma = cubic_splines_evaluation.evaluate(is_complex)
        calibration_sensitivity = self.get_calibration_sensitivity(sigma)
        loss = calibration_sensitivity.evaluate(is_complex)
        return loss

    def evaluate_f(self) :
        # To Do: reuse the same idea throughout the project, think of including it in the black box abstract class
        cubic_splines_construction = Cubic_splines_construction(self.S_p, self.sigma_p)
        cubic_splines_evaluation = Cubic_splines_evaluation(self.S, cubic_splines_construction)
        sigma = cubic_splines_evaluation.evaluate()
        calibration_sensitivity = self.get_calibration_sensitivity(sigma)
        # loss = calibration_sensitivity.evaluate()
        return cubic_splines_evaluation, calibration_sensitivity

    def forward(self, diff_u) :
        assert len(diff_u) == 1
        diff_sigma_p = diff_u[0]
        diff_S = np.zeros(self.S.shape)
        cubic_splines_construction = Cubic_splines_construction(self.S_p, self.sigma_p)
        cubic_splines_evaluation = Cubic_splines_evaluation(self.S, cubic_splines_construction)
        diff_sigma = cubic_splines_evaluation.forward([diff_S, diff_sigma_p])
        sigma = cubic_splines_evaluation.evaluate()
        calibration_sensitivity = self.get_calibration_sensitivity(sigma)
        diff_r = 0  # we assume r is fixed
        diff_loss = calibration_sensitivity.forward([diff_sigma, diff_r])
        # loss = calibration_sensitivity_adjoint.evaluate()
        return diff_loss

    def reverse(self, loss_bar=1) :
        # forward pass
        cubic_splines_evaluation, calibration_sensitivity = self.evaluate_f()
        # backward pass
        sigma_bar, r_bar = calibration_sensitivity.reverse(loss_bar)
        S_bar, sigma_p_bar = cubic_splines_evaluation.reverse(sigma_bar)  # ignore r_bar because r is constant
        return sigma_p_bar  # ignore S_bar because we assume that S is constant

    def validate_reverse(self, diff_u, qoi_bar) :
        # forward mode
        diff_qoi = [self.forward(diff_u)]
        # reverse mode
        u_bar = [self.reverse(qoi_bar)]
        b, err = check_forward_reverse_mode_identity(diff_u, u_bar, [], [], diff_qoi, [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)

    def optimize(self, nbr_epoch: int, lr: float, is_printed: bool = False):
        for epoch in range(nbr_epoch) :
            sigma_p_bar = self.reverse()
            if is_printed:
                loss = self.evaluate()  # TO DO: optimize it when doing forward and then backward
                print("Epoch:", epoch,
                      " Loss:", np.round(loss, 10),
                      " sigma_p_bar:", np.round(sigma_p_bar, 5))
            # gradient descent step
            sigma_p_new = self.sigma_p - lr * sigma_p_bar
            u_new = [sigma_p_new]
            self.set_u(u_new)
        if is_printed:
            print("Resulting sigma_p is: ",  self.sigma_p)
