from black_box import Black_box
import numpy as np
from auxiliary_functions import check_forward_reverse_mode_identity
from splines.Cubic_splines_construction import Cubic_splines_construction


class Cubic_splines_evaluation(Black_box):
    def __init__(self, S: np.ndarray, cubic_splines_construction: Cubic_splines_construction) :
        self.S = S
        self.cubic_splines_construction = cubic_splines_construction

    def __repr__(self) :
        return "Cubic splines evaluation: S_k=" + str(self.S) + " with Cubic Splines Construction=" + str(self.sigma_p)

    def copy(self) :
        return Cubic_splines_evaluation(self.S, self.cubic_splines_construction.copy())

    def set_u(self, u: list) :
        assert type(u) == list and len(u) == 2
        self.S, self.cubic_splines_construction.sigma_p = u

    def get_u(self) -> list:
        return [self.S, self.cubic_splines_construction.sigma_p]

    def evaluate(self, is_complex=False) -> float:
        sigma = np.zeros(len(self.S), dtype=complex) if is_complex else np.zeros(len(self.S))
        S_p = self.cubic_splines_construction.S_p
        sigma_p = self.cubic_splines_construction.sigma_p
        sigma_p_double_prime = self.cubic_splines_construction.evaluate(is_complex=is_complex)
        for j in range(len(self.S)) :
            S_j = min(max(self.S[j], min(S_p)), max(S_p))
            p = np.searchsorted(S_p, S_j)
            if p == 0:
                sigma[j] = sigma_p[p]
            else:
                h_p = S_p[p] - S_p[p - 1]
                summand1 = sigma_p_double_prime[p - 1] * (S_p[p] - S_j) ** 3 / (6 * h_p)
                summand2 = sigma_p_double_prime[p] * (S_j - S_p[p - 1]) ** 3 / (6 * h_p)
                summand3 = (sigma_p[p - 1] - (sigma_p_double_prime[p - 1] * h_p ** 2) / 6) * (S_p[p] - S_j) / h_p
                summand4 = (sigma_p[p] - (sigma_p_double_prime[p] * h_p ** 2) / 6) * (S_j - S_p[p - 1]) / h_p
                sigma[j] = summand1 + summand2 + summand3 + summand4
        return sigma

    def forward(self, diff_u):
        assert len(diff_u) == 2
        diff_S, diff_sigma_p = diff_u
        assert type(diff_S) == np.ndarray and type(diff_sigma_p) == np.ndarray
        diff_sigma_p_double_prime = self.cubic_splines_construction.forward([diff_sigma_p])
        diff_sigma = np.zeros(len(self.S))
        S_p = self.cubic_splines_construction.S_p
        sigma_p = self.cubic_splines_construction.sigma_p
        sigma_p_double_prime = self.cubic_splines_construction.evaluate()
        for j in range(len(self.S)) :
            S_j = min(max(self.S[j], min(S_p)), max(S_p))
            p = np.searchsorted(S_p, S_j)
            h_p = S_p[p] - S_p[p - 1]
            diff_summand1 = diff_sigma_p_double_prime[p - 1] * (S_p[p] - S_j) ** 3 / (6 * h_p) \
                            - sigma_p_double_prime[p - 1] * (S_p[p] - S_j) ** 2 / (2 * h_p) * diff_S[j]
            diff_summand2 = diff_sigma_p_double_prime[p] * (S_j - S_p[p - 1]) ** 3 / (6 * h_p) \
                            + sigma_p_double_prime[p] * (S_j - S_p[p - 1]) ** 2 / (2 * h_p) * diff_S[j]
            diff_summand3 = (diff_sigma_p[p - 1] - (diff_sigma_p_double_prime[p - 1] * h_p ** 2) / 6) * (
                        S_p[p] - S_j) / h_p \
                            - (sigma_p[p - 1] - (sigma_p_double_prime[p - 1] * h_p ** 2) / 6) * diff_S[j] / h_p
            diff_summand4 = (diff_sigma_p[p] - (diff_sigma_p_double_prime[p] * h_p ** 2) / 6) * (
                        S_j - S_p[p - 1]) / h_p \
                            + (sigma_p[p] - (sigma_p_double_prime[p] * h_p ** 2) / 6) * diff_S[j] / h_p
            diff_sigma[j] = diff_summand1 + diff_summand2 + diff_summand3 + diff_summand4
        return diff_sigma

    def reverse(self, sigma_bar: np.ndarray):
        assert len(sigma_bar) == len(self.S)
        S_p = self.cubic_splines_construction.S_p
        sigma_p = self.cubic_splines_construction.sigma_p
        sigma_p_double_prime = self.cubic_splines_construction.evaluate()
        S_bar = np.zeros(len(self.S))
        sigma_p_bar = np.zeros(len(S_p))
        sigma_p_double_prime_bar = np.zeros(len(S_p))
        for j in range(len(self.S)) :
            S_j = min(max(self.S[j], min(S_p)), max(S_p))
            p = np.searchsorted(S_p, S_j)
            h_p = S_p[p] - S_p[p - 1]
            sigma_p_bar[p] = sigma_p_bar[p] + sigma_bar[j] * (S_j - S_p[p - 1]) / h_p
            sigma_p_bar[p - 1] = sigma_p_bar[p - 1] + sigma_bar[j] * (S_p[p] - S_j) / h_p
            sigma_p_double_prime_bar[p] = sigma_p_double_prime_bar[p] + \
                                          sigma_bar[j] * ((S_j - S_p[p - 1]) ** 3 / (6 * h_p) - h_p / 6 * (
                        S_j - S_p[p - 1]))
            sigma_p_double_prime_bar[p - 1] = sigma_p_double_prime_bar[p - 1] + \
                                              sigma_bar[j] * ((S_p[p] - S_j) ** 3 / (6 * h_p) - h_p / 6 * (
                        S_p[p] - S_j))
            summand1 = - sigma_p_double_prime[p - 1] * (S_p[p] - S_j) ** 2 / (2 * h_p)
            summand2 = sigma_p_double_prime[p] * (S_j - S_p[p - 1]) ** 2 / (2 * h_p)
            summand3 = - (sigma_p[p - 1] - (sigma_p_double_prime[p - 1] * h_p ** 2) / 6) * 1 / h_p
            summand4 = (sigma_p[p] - (sigma_p_double_prime[p] * h_p ** 2) / 6) * 1 / h_p
            S_bar[j] = sigma_bar[j] * (summand1 + summand2 + summand3 + summand4)
        delta_sigma_p_bar = self.cubic_splines_construction.reverse(sigma_p_double_prime_bar)
        sigma_p_bar = sigma_p_bar + delta_sigma_p_bar
        return S_bar, sigma_p_bar

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = list(self.reverse(qoi_bar))
        b, err = check_forward_reverse_mode_identity(diff_u, u_bar, [], [], diff_qoi, qoi_bar, [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)
