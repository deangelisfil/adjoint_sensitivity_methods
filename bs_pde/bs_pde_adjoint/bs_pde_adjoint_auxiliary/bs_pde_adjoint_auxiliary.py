import numpy as np
from black_box import Black_box
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.forward_mode import bs_pde_adjoint_auxiliary_forward
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.reverse_mode import bs_pde_adjoint_auxiliary_reverse
from bs_pde.bs_pde_adjoint.bs_pde_adjoint_auxiliary.forward_pass import bs_pde_adjoint_auxiliary
from auxiliary_functions import check_forward_reverse_mode_identity


class Bs_pde_adjoint_auxiliary(Black_box) :
    def __init__(self, B: np.ndarray) :
        self.B = B

    def copy(self) :
        return Bs_pde_adjoint_auxiliary(self.B)

    def get_u(self) -> list :
        return [self.B]

    def set_u(self, u: list) :
        assert len(u) == 1
        self.B = u[0]

    def evaluate(self, is_complex=False) :
        return bs_pde_adjoint_auxiliary(self.B)

    def forward(self, diff_u) :
        assert len(diff_u) == 1
        diff_B = diff_u[0]
        return bs_pde_adjoint_auxiliary_forward(self.B, diff_B)

    def reverse(self, p_bar) :
        return bs_pde_adjoint_auxiliary_reverse(self.B, p_bar)

    def validate_reverse(self, diff_u, qoi_bar) :
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([], [], diff_u, [u_bar], [diff_qoi], [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)
