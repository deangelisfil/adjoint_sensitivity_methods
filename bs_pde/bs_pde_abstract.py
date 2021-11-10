from abc import abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity
from black_box import Black_box
import numpy as np

class Bs_pde_abstract(Black_box):
    """Note that classes derived from Bs_pde_abstract take S0 as an input
    and build a grid around S0 reaching [0,2*S0]"""
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

    def set_u(self, u: list):
        assert(len(u)==3)
        S0, sigma, r = u
        self.S0 = S0
        self.sigma = sigma
        self.r = r

    def get_u(self) -> list:
        return [self.S0, self.sigma, self.r]

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([diff_u], [u_bar], [], [], [diff_qoi], [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)
