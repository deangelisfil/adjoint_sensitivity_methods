from bs_pde_adjoint.forward_mode import bs_pde_adjoint_forward
from bs_pde_adjoint.reverse_mode import bs_pde_adjoint_reverse
from bs_pde_adjoint.forward_pass import bs_pde_adjoint
from black_blox import Black_box
from auxiliary_functions import check_forward_reverse_mode_identity
from bs_pde_abstract import Bs_pde_abstract

class Bs_pde_adjoint(Bs_pde_abstract):
    def __init__(self, S0, sigma, r) :
        self.S0 = S0
        self.sigma = sigma
        self.r = r

    def __repr__(self):
        return "Adjoint BS PDE: S0=" + str(self.S0) + ", sigma=" + str(self.sigma) + ", r=" + str(self.r)

    def copy(self):
        return Bs_pde_adjoint(self.S0, self.sigma, self.r)

    def evaluate(self, is_complex=False) -> float:
        return bs_pde_adjoint(self.S0, self.sigma, self.r, is_complex)

    def forward(self, diff_u):
        diff_S0, diff_sigma, diff_r = diff_u
        return bs_pde_adjoint_forward(self.S0, self.sigma, self.r, diff_S0, diff_sigma, diff_r)

    def reverse(self, qoi_bar=1):
        return bs_pde_adjoint_reverse(self.S0, self.sigma, self.r, qoi_bar)



