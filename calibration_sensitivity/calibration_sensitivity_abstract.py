from abc import abstractmethod
from black_box import Black_box
from auxiliary_functions import check_forward_reverse_mode_identity
import numpy as np

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

    def set_u(self, u: list) :
        assert len(u) == 2
        self.sigma = u[0]
        self.r = u[1]

    def get_u(self) -> list:
        return [self.sigma, self.r]

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = [self.forward(diff_u)]
        # reverse mode
        u_bar = list(self.reverse(qoi_bar))
        b, err = check_forward_reverse_mode_identity(diff_u, u_bar, [], [], diff_qoi, [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)

    def optimize(self, nbr_epoch: int, lr: float, is_printed: bool = False):
        for epoch in range(nbr_epoch) :
            sigma_bar, r_bar = self.reverse()
            if is_printed:
                loss = self.evaluate()  # TO DO: optimize it when doing forward and then backward
                print("Epoch:", epoch,
                      " Loss:", np.round(loss, 10),
                      " sigma_bar:", np.round(sigma_bar, 5),
                      " r_bar:", np.round(r_bar, 5))
            # gradient descent step
            sigma_new = self.sigma - lr * sigma_bar
            r_new = self.r - lr * r_bar
            # update theta = {sigma, r}
            u_new = [sigma_new, r_new]
            self.set_u(u_new)

