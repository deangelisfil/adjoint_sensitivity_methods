from black_box import Black_box
import numpy as np
from auxiliary_functions import check_forward_reverse_mode_identity


class Cubic_splines_construction(Black_box):
    def __init__(self, S_p: np.ndarray, sigma_p: np.ndarray) :
        assert S_p.shape == sigma_p.shape
        assert (np.sort(S_p) == S_p).all()
        self.S_p = S_p
        self.sigma_p = sigma_p

    def __repr__(self):
        return "Cubic splines construction: S_p=" + str(self.S_p) + ", sigma_p=" + str(self.sigma_p)

    def copy(self) :
        return Cubic_splines_construction(self.S_p, self.sigma_p)

    def get_u(self) -> list:
        return [self.sigma_p]

    def set_u(self, u: list):
        assert len(u) == 1
        self.sigma_p = np.array(u[0])

    def construct_A_B(self):
        h = np.diff(self.S_p)
        mu = np.append(np.insert(h[:-1] / (h[:-1] + h[1:]), 0, 0), 1)
        centre_A = 2 * np.ones(self.S_p.shape)
        low_A = mu[1:]
        up_A = 1 - mu[:-1]
        A = np.diag(low_A, -1) + np.diag(centre_A, 0) + np.diag(up_A, 1)

        centre_B = np.append(np.insert(-1 / (h[:-1] * h[1 :]), 0, -1 / h[0] ** 2), 1/h[-1] ** 2)

        low_B = np.append(1 / (h[:-1] * (h[:-1] + h[1:])), -1/h[-1]**2)
        up_B = np.insert(1 / (h[1:] * (h[:-1] + h[1:])), 0, 1/h[0]**2)
        B = 6 * (np.diag(low_B, -1) + np.diag(centre_B, 0) + np.diag(up_B, 1))
        return A, B

    def evaluate(self, is_complex=False) -> float:
        # complex behavior is the same
        A, B = self.construct_A_B()
        sigma_p_double_prime = np.linalg.inv(A) @ (B @ self.sigma_p)
        return sigma_p_double_prime

    def forward(self, diff_u):
        assert isinstance(diff_u, list) and len(diff_u) == 1
        diff_sigma_p = diff_u[0]
        A, B = self.construct_A_B()
        diff_sigma_p_double_prime = np.linalg.inv(A) @ (B @ diff_sigma_p)
        return diff_sigma_p_double_prime

    def reverse(self, sigma_p_double_prime_bar: np.ndarray):
        # assumes that sigma_p_bar starts at zero and computes the extra dependence of sigma_p_bar due
        # to the way in which sigma_p_double_prime is computed
        assert sigma_p_double_prime_bar.shape == self.sigma_p.shape
        A, B = self.construct_A_B()
        sigma_p_bar = B.transpose() @ (np.linalg.inv(A.transpose()) @ sigma_p_double_prime_bar)
        return sigma_p_bar

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity(diff_u, [u_bar], [], [], diff_qoi, qoi_bar, [], [])
        print("The forward/ reverse mode identity holds:", b, "the error is:", err)


