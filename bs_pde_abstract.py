from abc import ABC, abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity
from black_blox import Black_box

class Bs_pde_abstract(Black_box):
    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar=1):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def copy(self):
        pass

    def validate(self, diff_u, qoi_bar):
        # bumping
        epsilon = 10e-7
        bs_pde_plus = self.copy()
        bs_pde_plus.S0 = self.S0 + epsilon
        bs_pde_minus = self.copy()
        bs_pde_minus.S0 = self.S0 - epsilon
        delta_bumping = (bs_pde_plus.evaluate() - bs_pde_minus.evaluate()) / (2 * epsilon)

        # complex variable trick
        epsilon = 10e-20
        bs_pde_complex = self.copy()
        bs_pde_complex.S0 = self.S0 + epsilon * 1j
        delta_complex_trick = bs_pde_complex.evaluate(is_complex=True).imag / epsilon

        # forward mode
        qoi, delta = self.forward([1,0,0]) # since we compare this value with delta, choose diff_u = [1,0,0]
        qoi, diff_qoi = self.forward(diff_u)

        # reverse
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([diff_u], [u_bar], [], [], [diff_qoi], [qoi_bar], [], [])

        print("Difference between the complex variable trick and bumping is: ",
              abs(delta_bumping - delta_complex_trick))
        print("Difference of delta between the forward mode and complex variable trick is:",
              abs(delta_complex_trick - delta))
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)