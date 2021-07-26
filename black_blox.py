from abc import ABC, abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity
import numpy as np

class Black_box(ABC):

    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def set_u(self, u: list):
        pass

    @abstractmethod
    def get_u(self) -> list:
        pass

    def validate_forward(self, idx):
        """
        validates l = [0,..0], l[idx]=1 in the forward mode
        """
        # bumping
        epsilon = 10e-7
        u = self.get_u()
        assert type(idx) == int and idx <= len(u)

        u_increment = np.zeros(len(u));
        u_increment[idx] = 1
        black_box_plus = self.copy()
        u_plus = u.copy();
        u_plus[idx] = u_plus[idx] + epsilon
        black_box_plus.set_u(u_plus)
        black_box_minus = self.copy()
        u_minus = u.copy();
        u_minus[idx] = u_minus[idx] - epsilon
        black_box_minus.set_u(u_minus)
        diff_qoi_bumping = (black_box_plus.evaluate() - black_box_minus.evaluate()) / (2 * epsilon)

        # complex variable trick
        epsilon = 10e-20
        black_box_complex = self.copy()
        u_complex = u.copy()
        if type(u_complex) is np.ndarray:
            u_complex = u_complex.astype(complex)
        u_complex[idx] = u_complex[idx] + epsilon * 1j
        black_box_complex.set_u(u_complex)
        diff_qoi_complex_trick = black_box_complex.evaluate(is_complex=True).imag / epsilon

        # forward mode
        diff_u = len(u) * [0]
        diff_u[idx] = 1
        diff_qoi = self.forward(diff_u)

        err_complex = abs(diff_qoi_bumping - diff_qoi_complex_trick)
        err_forward = abs(diff_qoi_complex_trick - diff_qoi)

        if type(err_complex) is np.ndarray and type(err_forward) is np.ndarray:
            n = len(err_complex)
            rel_err_complex = np.zeros(n)
            rel_err_forward = np.zeros(n)
            for i in range(n):
                rel_err_complex[i] = err_complex[i] / 10e-16 if abs(diff_qoi_complex_trick[i]) < 10e-16 else err_complex[i] / diff_qoi_complex_trick[i]
                rel_err_forward[i] = err_forward[i] / 10e-16 if abs(diff_qoi[i]) < 10e-16 else err_forward[i] / diff_qoi[i]
        else:
            rel_err_complex = err_complex / 10e-16 if abs(diff_qoi_complex_trick) < 10e-16 else err_complex / diff_qoi_complex_trick
            rel_err_forward = err_forward / 10e-16 if abs(diff_qoi) < 10e-16 else err_forward / diff_qoi

        print("Difference between the complex variable trick and bumping is: ", rel_err_complex)
        print("Difference of delta between the forward mode and complex variable trick is:", rel_err_forward)

    @abstractmethod
    def validate_reverse(self, diff_u, qoi_bar):
        pass

    def validate(self, idx_forward_validation, diff_u, qoi_bar):
        self.validate_forward(idx_forward_validation)
        self.validate_reverse(diff_u, qoi_bar)
