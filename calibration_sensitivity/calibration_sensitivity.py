from calibration_sensitivity.forward_mode import calibration_sensitivity_forward
from calibration_sensitivity.reverse_mode import calibration_sensitivity_reverse
from calibration_sensitivity.forward_pass import calibration_sensitivity
from bs_pde_abstract import Bs_pde_abstract

class Calibration_sensitivity(Bs_pde_abstract):
    def __init__(self, S0: float, sigma: float, r: float, option_list: list,
                 loss_fn: callable, diff_loss_fn: callable) :
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.option_list = option_list
        self.loss_fn = loss_fn
        self.diff_loss_fn = diff_loss_fn

    def __repr__(self):
        return "Calibration sensitivity: S0=" + str(self.S0) + ", sigma=" + str(self.sigma) + ", r=" + str(self.r) + \
               ", K: " + str(len(self.option_list))

    def copy(self) :
        return Calibration_sensitivity(self.S0, self.sigma, self.r, self.option_list, self.loss_fn, self.diff_loss_fn)

    def evaluate(self, is_complex=False) -> float:
        return calibration_sensitivity(self.S0, self.sigma, self.r, self.option_list, self.loss_fn)

    def forward(self, diff_u):
        diff_S0, diff_sigma, diff_r = diff_u
        return calibration_sensitivity_forward(self.S0, self.sigma, self.r, diff_S0, diff_sigma, diff_r, self.option_list,
                                               self.loss_fn, self.diff_loss_fn)

    def reverse(self, loss_bar=1):
        return calibration_sensitivity_reverse(self.S0, self.sigma, self.r, self.option_list, self.diff_loss_fn, loss_bar)