from calibration_sensitivity.calibration_sensitivity_abstract import Calibration_sensitivity_abstract
from calibration_sensitivity.calibration_loss import Calibration_loss
import numpy as np
from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from parameters import *
from bs_pde.bs_pde_standard.bs_pde_standard_auxiliary import Bs_pde_standard_auxiliary


class Calibration_sensitivity_standard(Calibration_sensitivity_abstract) :
    def __init__(self, S: np.ndarray, sigma: float, r: float, option_list: list,
                 loss: Calibration_loss, american: bool = False) :
        self.S = S
        self.sigma = sigma
        self.r = r
        self.option_list = option_list
        self.loss = loss
        self.american = american

    def __repr__(self) :
        return "Calibration sensitivity: S=" + str(self.S) + ", sigma=" + str(self.sigma) + ", r=" + str(self.r) + \
               ", K=" + str(len(self.option_list))

    def copy(self) :
        return Calibration_sensitivity_standard(self.S, self.sigma, self.r, self.option_list, self.loss, self.american)

    def evaluate(self, is_complex=False) -> float :
        B_construction = B_construction_time_invariant(self.S, self.sigma, self.r, delta_t, delta_S)
        B = B_construction.evaluate()
        K = len(self.option_list)
        f = np.array(list(map(lambda x : x.evaluate(self.S), self.option_list)))
        P_model = np.zeros(K, dtype=complex) if is_complex else np.zeros(K)
        for k in range(K) :
            bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f[k], american = self.american)
            P_model[k] = bs_pde_standard_auxiliary.evaluate(is_complex=is_complex)
        loss = self.loss.evaluate(P_model)
        return loss

    def evaluate_f(self) :
        B_construction = B_construction_time_invariant(self.S, self.sigma, self.r, delta_t, delta_S)
        B = B_construction.evaluate()
        K = len(self.option_list)
        f = np.array(list(map(lambda x : x.evaluate(self.S), self.option_list)))
        P_model = np.zeros(K)
        for k in range(K) :
            bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f[k], american = self.american)
            P_model[k] = bs_pde_standard_auxiliary.evaluate()
        # loss = self.loss.evaluate(P_model)
        return B, f, P_model

    def forward(self, diff_u) :
        assert len(diff_u) == 2
        diff_sigma, diff_r = diff_u
        diff_S = np.zeros(len(self.S))  # view S as fixed
        B_construction = B_construction_time_invariant(self.S, self.sigma, self.r, delta_t, delta_S)
        diff_B = B_construction.forward([diff_S, diff_sigma, diff_r])
        B = B_construction.evaluate()
        K = len(self.option_list)
        f = np.array(list(map(lambda x : x.evaluate(self.S), self.option_list)))
        diff_f = np.zeros(f.shape)
        diff_P_model = np.zeros(K);
        P_model = np.zeros(K)
        for k in range(K) :
            bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f[k], american = self.american)
            diff_P_model[k] = bs_pde_standard_auxiliary.forward([diff_B, diff_f[k]])
            P_model[k] = bs_pde_standard_auxiliary.evaluate()
        diff_loss = np.dot(self.loss.diff_evaluate(P_model), diff_P_model)
        # loss = self.loss.evaluate(P_model)
        return diff_loss

    def reverse(self, loss_bar=1) :
        # forward pass
        B, f, P_model = self.evaluate_f()
        # backward pass
        P_model_bar = self.loss.diff_evaluate(P_model) * loss_bar
        K = len(self.option_list)
        B_bar = np.zeros(B.shape)
        # f_bar = np.zeros(f.shape)
        for k in range(K) :
            bs_pde_standard_auxiliary = Bs_pde_standard_auxiliary(B, f[k], american = self.american)
            B_bar_delta, f_bar_delta = bs_pde_standard_auxiliary.reverse(P_model_bar[k])
            B_bar += B_bar_delta
            # f_bar += f_bar_delta # ignore f_bar as we ignore S_bar
        B_construction = B_construction_time_invariant(self.S, self.sigma, self.r, delta_t, delta_S)
        S_bar, sigma_bar, r_bar = B_construction.reverse(B_bar)
        # ignore S_bar
        return sigma_bar, r_bar
