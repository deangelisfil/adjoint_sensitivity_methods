from abc import abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity
from black_blox import Black_box
import numpy as np

class Bs_pde_abstract(Black_box):

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

    def set_u(self, u: np.ndarray):
        assert(len(u)==3)
        S0, sigma, r = u
        self.S0 = S0
        self.sigma = sigma
        self.r = r

    def get_u(self):
        return np.array([self.S0, self.sigma, self.r])

    def validate_forward(self, idx):
        """
        validates l = [0,0,0], l[idx]=1 in the forward mode
        """
        assert type(idx) == int and idx <= 2
        # bumping
        epsilon = 10e-7
        u_increment = np.zeros(3); u_increment[idx]=1
        bs_pde_plus = self.copy()
        bs_pde_plus.set_u(self.get_u() + u_increment * epsilon)
        bs_pde_minus = self.copy()
        bs_pde_minus.set_u(self.get_u() - u_increment * epsilon)
        delta_bumping = (bs_pde_plus.evaluate() - bs_pde_minus.evaluate()) / (2 * epsilon)

        # complex variable trick
        epsilon = 10e-20
        bs_pde_complex = self.copy()
        bs_pde_complex.set_u(self.get_u() + u_increment * epsilon * 1j)
        delta_complex_trick = bs_pde_complex.evaluate(is_complex=True).imag / epsilon

        # forward mode
        u = [0,0,0]; u[idx] = 1
        qoi, delta = self.forward(u)

        err_complex = delta_bumping - delta_complex_trick
        err_forward = delta_complex_trick - delta
        rel_err_complex = err_complex/10e-16 if delta_complex_trick < 10e-16 else err_complex/delta_complex_trick
        rel_err_forward = err_forward/10e-16 if delta < 10e-16 else err_forward/delta

        print("Difference between the complex variable trick and bumping is: ", rel_err_complex)
        print("Difference of delta between the forward mode and complex variable trick is:", rel_err_forward)

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        qoi, diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([diff_u], [u_bar], [], [], [diff_qoi], [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)