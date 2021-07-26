import numpy as np



def B_construction_time_invariant_f(S, sigma, r, delta_t, delta_S) :
    diff = delta_t / delta_S
    diff2 = delta_t / delta_S ** 2
    a = S ** 2 * sigma ** 2 * diff2 / 2
    b = S * r * diff / 2
    low = a[1 :] - b[1 :]
    low[-1] = - r * S[-1] * diff
    centre = 1 - r * delta_t - 2 * a
    centre[-1] = 1 - r * delta_t + r * S[-1] * diff
    up = a[:-1] + b[:-1]
    B = np.diag(low, -1) + np.diag(centre, 0) + np.diag(up, 1)
    return B


def B_construction_time_invariant_forward(S, sigma, r, delta_t, delta_S, diff_S, diff_sigma, diff_r) :
    diff = delta_t / delta_S
    diff2 = delta_t / delta_S ** 2
    a = S ** 2 * (sigma * diff_sigma * diff2) + S * diff_S * (sigma ** 2 * diff2)
    b = S * diff_r * diff / 2 + diff_S * r * diff / 2
    low = a[1 :] - b[1 :]
    low[-1] = - diff_r * S[-1] * diff - r * diff_S[-1] * diff
    centre = -diff_r * delta_t - 2 * a
    centre[-1] = - diff_r * delta_t + diff_r * S[-1] * diff + r * diff_S[-1] * diff
    up = a[:-1] + b[:-1]
    diff_B = np.diag(low, -1) + np.diag(centre, 0) + np.diag(up, 1)
    return diff_B


def B_construction_time_invariant_reverse(S, sigma, r, delta_t, delta_S, B_bar, S_bar) :
    """
    :param S_bar: increases it. If there is no S_bar before, set it to 0.
    """
    diff = delta_t / delta_S
    diff2 = delta_t / delta_S ** 2
    d = S.size
    if type(sigma) is np.ndarray:
        sigma_bar = np.zeros(len(sigma))
    else:
        sigma_bar = 0
    r_bar = (-delta_t + diff * S[-1]) * B_bar[-1][-1] - S[-1] * diff * B_bar[-1][-2]
    S_bar[-1] = S_bar[-1] + r * diff * (B_bar[-1][-1] - B_bar[-1][-2])
    for j in reversed(range(d - 1)) :
        if j == 0 :
            # B_bar[j-1][j] does not count
            a_bar = - 2 * B_bar[j][j] + B_bar[j][j + 1]
            b_bar = B_bar[j][j + 1]
        else :
            a_bar = B_bar[j][j - 1] - 2 * B_bar[j][j] + B_bar[j][j + 1]
            b_bar = - B_bar[j][j - 1] + B_bar[j][j + 1]
        if type(sigma) is np.ndarray:
            sigma_bar[j] = sigma_bar[j] + sigma[j] * S[j] ** 2 * diff2 * a_bar
            S_bar[j] = S_bar[j] + sigma[j] ** 2 * S[j] * diff2 * a_bar + r / 2 * diff * b_bar
        else:
            sigma_bar = sigma_bar + sigma * S[j] ** 2 * diff2 * a_bar
            S_bar[j] = S_bar[j] + sigma ** 2 * S[j] * diff2 * a_bar + r / 2 * diff * b_bar
        r_bar = r_bar + S[j] / 2 * diff * b_bar - delta_t * B_bar[j][j]
    return S_bar, sigma_bar, r_bar