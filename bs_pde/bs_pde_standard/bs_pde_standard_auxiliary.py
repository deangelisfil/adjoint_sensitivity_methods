from black_box import Black_box
import numpy as np
from parameters import grid
from auxiliary_functions import maximum
from auxiliary_functions import check_forward_reverse_mode_identity
from grid import Grid

class Bs_pde_standard_auxiliary(Black_box) :
    def __init__(self, B: np.ndarray,
                 f: np.ndarray,
                 grid_local: Grid = grid,
                 american: bool = False) :
        self.B = B
        self.f = f
        self.grid = grid_local
        self.american = american

    def copy(self) :
        return Bs_pde_standard_auxiliary(self.B, self.f, self.grid, self.american)

    def get_u(self) -> list :
        return [self.B, self.f]

    def set_u(self, u: list) :
        assert len(u) == 2
        self.B = u[0]
        self.f = u[1]

    def evaluate(self, is_complex=False) :
        u = np.copy(self.f)
        for n in reversed(range(self.grid.N)) :
            u = self.B.dot(u)
            if self.american :
                u = maximum(u, self.f, is_complex)
        qoi = u[self.grid.J]
        return qoi

    def evaluate_f(self) :
        u = np.copy(self.f)
        u_all_list = [u]
        u_hat_all_list = []
        for n in reversed(range(self.grid.N)) :
            u = self.B.dot(u)
            if self.american :
                u_hat_all_list.append(u)
                u = maximum(u, self.f, is_complex=False)
            u_all_list.append(u)
        qoi = u[self.grid.J]
        return qoi, list(reversed(u_all_list)), list(reversed(u_hat_all_list))

    def forward(self, diff_u_list) :
        assert len(diff_u_list) == 2
        diff_B = diff_u_list[0]
        diff_f = diff_u_list[1]
        diff_u = np.copy(diff_f)  # attention notation: this diff_u refers to \dot{u_N}
        u = np.copy(self.f)
        for n in reversed(range(self.grid.N)) :
            diff_u = self.B.dot(diff_u) + diff_B.dot(u)
            u = self.B.dot(u)
            if self.american :
                # holding  = heaviside_close(u-option.payoff(S), 1)
                holding = np.heaviside(u - self.f, 1)
                diff_u = diff_u * holding + diff_f * (1 - holding)
                u = maximum(u, self.f, is_complex=False)
        diff_qoi = diff_u[self.grid.J]
        # qoi = u[J]
        return diff_qoi

    def reverse(self, qoi_bar=1) :
        # forward pass
        qoi, u_all_list, u_hat_all_list = self.evaluate_f()
        # backward pass
        assert (not (self.american is True and not u_hat_all_list)), "American option with empty u_hat list"
        d = 2 * self.grid.J + 1
        B_bar = np.zeros((d, d))
        u_bar = np.zeros(d)
        u_bar[self.grid.J] = qoi_bar
        f_bar = 0
        B_transpose = self.B.transpose()
        for n in range(self.grid.N) :
            if self.american :
                holding = np.heaviside(u_hat_all_list[n] - self.f, 1)
                # holding = heaviside_close(u_hat_all_list[n] - option.payoff(S), 1)
                # print("holding: ", holding)
                u_hat_bar = u_bar * holding
                f_bar = f_bar + u_bar * (1 - holding)
                B_bar = B_bar + np.outer(u_hat_bar, u_all_list[n + 1])
                u_bar = B_transpose.dot(u_hat_bar)
            else :
                B_bar = B_bar + np.outer(u_bar, u_all_list[n + 1])
                u_bar = B_transpose.dot(u_bar)
        f_bar = f_bar + u_bar
        return B_bar, f_bar

    def validate_reverse(self, diff_u, qoi_bar):
        # forward mode
        diff_qoi = self.forward(diff_u)
        # reverse mode
        u_bar = self.reverse(qoi_bar)
        b, err = check_forward_reverse_mode_identity([diff_u[1]], [u_bar[1]], [diff_u[0]], [u_bar[0]],
                                                     [diff_qoi], [qoi_bar], [], [])
        print("The forward/ reverse mode identiy holds:", b, "the error is:", err)

