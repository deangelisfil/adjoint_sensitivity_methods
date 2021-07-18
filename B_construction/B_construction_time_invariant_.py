import numpy as np
from black_blox import Black_box
from B_construction.b_construction_time_invariant import *

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
        pass

    def validate_reverse(self, diff_u, qoi_bar):
        pass
