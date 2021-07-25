import numpy as np
from black_blox import Black_box
from B_construction.b_construction_time_invariant import *
from auxiliary_functions import check_forward_reverse_mode_identity

class B_construction_time_invariant(Black_box):
    def __init__(self, S: np.ndarray, sigma: float, r: float, delta_t: float, delta_S: float) :
        self.S = S
        self.sigma = sigma
        self.r = r
        self.delta_t = delta_t
        self.delta_S = delta_S

    def copy(self):
        return B_construction_time_invariant(self.S, self.sigma, self.r, self.delta_t, self.delta_S)

    def get_u(self):
        return [self.S, self.sigma, self.r]

    def set_u(self, u: list):
        assert(len(u)==3)
        S, sigma, r = u
        self.S = S
        self.sigma = sigma
        self.r = r

    def evaluate(self, is_complex=False) -> float:
        return B_construction_time_invariant_f(self.S, self.sigma, self.r, self.delta_t, self.delta_S)

    def forward(self, diff_u):
        diff_S, diff_sigma, diff_r = diff_u
        return B_construction_time_invariant_forward(self.S, self.sigma, self.r, self.delta_t, self.delta_S,
                                                     diff_S, diff_sigma, diff_r)
    def reverse(self, qoi_bar):
        S_bar = np.zeros(self.S.size)
        return B_construction_time_invariant_reverse(self.S, self.sigma, self.r, self.delta_t, self.delta_S,
                                                     qoi_bar, S_bar)

    def validate_forward(self, idx):
        #To do: have an idea on how to check this thing
        pass

    def validate_reverse(self, diff_u, B_bar):
        # forward mode
        diff_B = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(B_bar)
        b, err = check_forward_reverse_mode_identity(diff_u, list(u_bar), [], [], [], [], [diff_B], [B_bar])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)