from calibration_sensitivity.calibration_sensitivity_standard.forward_mode import calibration_sensitivity_standard_forward
from calibration_sensitivity.calibration_sensitivity_standard.reverse_mode import calibration_sensitivity_standard_reverse
from calibration_sensitivity.calibration_sensitivity_standard.forward_pass import calibration_sensitivity_standard
from calibration_sensitivity.calibration_abstract import Calibration_sensitivity_abstract
from calibration_sensitivity.calibration_loss import Calibration_loss

class Calibration_sensitivity_standard(Calibration_sensitivity_abstract):
    def __init__(self, S0: float, sigma: float, r: float, option_list: list,
                 loss: Calibration_loss, american: bool=False) :
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.option_list = option_list
        self.loss = loss
        self.american = american

    def __repr__(self):
        return "Calibration sensitivity: S0=" + str(self.S0) + ", sigma=" + str(self.sigma) + ", r=" + str(self.r) + \
               ", K=" + str(len(self.option_list))

    def copy(self) :
        return Calibration_sensitivity_standard(self.S0, self.sigma, self.r, self.option_list, self.loss, self.american)

    def evaluate(self, is_complex=False) -> float:
        return calibration_sensitivity_standard(self.S0, self.sigma, self.r, self.option_list, self.loss, self.american,
                                                is_complex)

    def forward(self, diff_u):
        diff_S0, diff_sigma, diff_r = diff_u
        return calibration_sensitivity_standard_forward(self.S0, self.sigma, self.r, diff_S0, diff_sigma, diff_r,
                                                        self.option_list, self.loss, self.american)

    def reverse(self, loss_bar=1):
        return calibration_sensitivity_standard_reverse(self.S0, self.sigma, self.r, self.option_list, self.loss,
                                                        self.american, loss_bar)