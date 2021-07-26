from abc import abstractmethod
from black_blox import Black_box
from auxiliary_functions import check_forward_reverse_mode_identity

class Calibration_sensitivity_abstract(Black_box):

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar=1):
        pass

    def get_u(self) -> list:
        return [self.sigma, self.r]

    def set_u(self, u: list) :
        assert len(u) == 2
        self.sigma = u[0]
        self.r = u[1]

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = [self.forward(diff_u)]
        # reverse mode
        u_bar = list(self.reverse(qoi_bar))
        b, err = check_forward_reverse_mode_identity(diff_u, u_bar, [], [], diff_qoi, [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)

