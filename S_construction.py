from black_box import Black_box
import numpy as np
from auxiliary_functions import check_forward_reverse_mode_identity
from grid import Grid

# To do: add the S_construction to all functions like I did with bs_pde_standard
# To do: check whether it makes sense to view delta_S as a variable of S0
# (and hence, take it into account for forward and reverse mode)


class S_construction(Black_box):
    def __init__(self, S0: np.array, grid: Grid):
        self.S0 = S0
        self.grid = grid

    def copy(self):
        return S_construction(self.S0, self.grid)

    def get_u(self) -> list :
        return [self.S0]

    def set_u(self, u: list) :
        assert len(u) == 1
        self.S0 = u[0]

    def evaluate(self, is_complex=False) -> float:
        S = np.array([self.S0 + j * self.grid.delta_S for j in range(-self.grid.J, self.grid.J + 1)])
        return S

    def forward(self, diff_u):
        assert len(diff_u) == 1
        diff_S0 = diff_u[0]
        return diff_S0 * np.ones(2*self.grid.J + 1)

    def reverse(self, qoi_bar):
        assert isinstance(qoi_bar, np.ndarray)
        return sum(qoi_bar)

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([diff_u], [u_bar], [], [],
                                                     [diff_qoi], [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)

