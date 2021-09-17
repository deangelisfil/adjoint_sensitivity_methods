from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from bs_pde.bs_pde_standard.forward_pass import bs_pde_standard_f
from function  import Function
from parameters import *

# reverse mode
def bs_pde_standard_auxiliary_b(B, qoi_bar=1) :
    d = 2 * J + 1
    u_bar = np.zeros(d)
    u_bar[J] = qoi_bar
    b_bar_all_list = []
    B_transpose = B.transpose()
    for n in range(N) :
        b_bar = u_bar
        u_bar = np.dot(B_transpose, u_bar)
        b_bar_all_list.append(b_bar)
    f_bar = u_bar
    return f_bar, b_bar_all_list


def bs_pde_standard_auxiliary_reverse(B, qoi_bar=1) :
    # forward pass
    # u_all_list = bs_pde_auxiliary_f(f, B)
    # backward pass
    f_bar, b_bar_all_list = bs_pde_standard_auxiliary_b(B, qoi_bar)
    return f_bar, b_bar_all_list


def bs_pde_standard_b(S: float,
                      sigma: float,
                      r: float,
                      option: Function,
                      B: np.ndarray,
                      u_all_list: list,
                      u_hat_all_list: list = [],
                      qoi_bar=1,
                      american=False) -> tuple:
    assert (not (american is True and not u_hat_all_list)), "American option with empty u_hat list"
    d = 2 * J + 1
    S_bar = np.zeros(d);
    B_bar = np.zeros((d, d))
    u_bar = np.zeros(d)
    u_bar[J] = qoi_bar
    B_transpose = B.transpose()
    for n in range(N) :
        if american :
            holding = np.heaviside(u_hat_all_list[n] - option.evaluate(S), 1)
            # holding = heaviside_close(u_hat_all_list[n] - option.payoff(S), 1)
            # print("holding: ", holding)
            u_hat_bar = u_bar * holding
            S_bar = S_bar + option.diff_evaluate(S) * u_bar * (1 - holding)
            B_bar = B_bar + np.outer(u_hat_bar, u_all_list[n + 1])
            u_bar = np.dot(B_transpose, u_hat_bar)
        else :
            B_bar = B_bar + np.outer(u_bar, u_all_list[n + 1])
            u_bar = np.dot(B_transpose, u_bar)
    B_construction = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    S_bar, sigma_bar, r_bar = B_construction.reverse(B_bar, S_bar = S_bar)
    S_bar = S_bar + option.diff_evaluate(S) * u_bar  # elementwise multiplication
    S0_bar = sum(S_bar)
    return S0_bar, sigma_bar, r_bar


def bs_pde_standard_reverse(S0, sigma, r, option, qoi_bar=1, american=False) :
    # forward pass
    qoi, S, B, u_all_list, u_hat_all_list = bs_pde_standard_f(S0, sigma, r, option, american)
    # backward pass
    S0_bar, sigma_bar, r_bar = bs_pde_standard_b(S, sigma, r, option, B, u_all_list, u_hat_all_list, qoi_bar, american)
    return S0_bar, sigma_bar, r_bar