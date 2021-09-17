import numpy as np


def maximum(arg1: np.array, arg2, is_complex: bool) -> np.array :
    """
    Elementwise maximum of two numpy arrays or of a numpy array with an integer.
    Defines the analytic extension of the maximum function in the complex case.
    If the entries are complex, return the entry with the biggest real part.
    If the real parts are equal, return the entry of the first argument.
    """
    if is_complex :
        res = []
        if type(arg2) == np.ndarray :
            assert arg1.shape == arg2.shape
            for el1, el2 in zip(arg1, arg2) :
                if el1.real >= el2.real or np.isclose(el1.real, 0, atol=1e-16) :
                    res.append(el1)
                else :
                    res.append(el2)
        else :
            # arg2 is integer
            assert type(arg2) == int
            for el in arg1 :
                if el.real >= arg2 :
                    res.append(el)
                else :
                    res.append(arg2)
        return np.array(res)
    else :
        return np.maximum(arg1, arg2)


def check_forward_reverse_mode_identity(diff_u_list=[], u_bar_list=[], diff_A_list=[], A_bar_list=[],
                                        diff_v_list=[], v_bar_list=[], diff_B_list=[], B_bar_list=[]) :
    # diff_u_list, u_bar_list is a list of input np arrays
    # diff_A_list, A_bar_list is a list of input nd arrays representing the input matrices
    # diff_v_list, v_bar_list is a list of output np arrays
    # diff_B_list, B_bar_list is a list of output nd arrays representing the output matrices
    # to do: assert that the dimension of u_list and u_bar_list as well as A_list and A_bar_list is the same
    sum_lhs = 0
    for diff_u, u_bar in zip(diff_u_list, u_bar_list) :
        sum_lhs += np.dot(u_bar, diff_u)
    for diff_A, A_bar in zip(diff_A_list, A_bar_list) :
        sum_lhs += np.trace(A_bar.transpose() @ diff_A)
    sum_rhs = 0
    for diff_v, v_bar in zip(diff_v_list, v_bar_list) :
        sum_rhs += np.dot(v_bar, diff_v)
    for diff_B, B_bar in zip(diff_B_list, B_bar_list) :
        sum_rhs += np.trace(B_bar.transpose() @ diff_B)
    err = abs(sum_lhs - sum_rhs)

    rel_err = err / 10e-15 if min(abs(sum_lhs), abs(sum_lhs)) < 10e-15 else err / min(abs(sum_lhs), abs(sum_lhs))
    return rel_err < 10e-15, rel_err
    # return err < 10e-15, err


def perturbe_scalar(scalar, type, epsilon) :
    assert type in {"plus", "minus", "complex"}
    if type == "plus" :
        scalar = scalar + epsilon
    elif type == "minus" :
        scalar = scalar - epsilon
    else :
        # type == "complex"
        scalar = scalar + epsilon * 1j
    return scalar

# def heaviside_close(x1, x2):
#     closeCheck = np.isclose(x1, np.zeros_like(x1), atol=1e-16)
#     heavisideBare = np.heaviside(x1, 0.0)
#     zeroVal = np.where(closeCheck, x2, 0.0)-np.where(closeCheck, heavisideBare, np.zeros_like(heavisideBare))
#     result = heavisideBare+zeroVal
#     return result
