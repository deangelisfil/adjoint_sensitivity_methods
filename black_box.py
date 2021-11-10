from abc import ABC, abstractmethod
from auxiliary_functions import perturbe_scalar
import numpy as np
import copy
from scipy import sparse


class Black_box(ABC) :
    @abstractmethod
    def forward(self, diff_u) :
        pass

    @abstractmethod
    def reverse(self, qoi_bar) :
        pass

    @abstractmethod
    def evaluate(self, is_complex=False) :
        pass

    @abstractmethod
    def set_u(self, u: list) :
        pass

    @abstractmethod
    def get_u(self) -> list :
        pass

    def zero_u(self):
        """creates an instance of u that has 0 scalars and arrays with 0 entries"""
        u = self.get_u()
        res = u.copy()
        for idx, el in enumerate(res):
            if isinstance(el, np.ndarray):
                res[idx] = np.zeros(el.shape)
            else:
                assert not hasattr(el, "__len__") # el is a scalar
                res[idx] = 0
        return res

    def perturbe_black_box(self, type, epsilon, idx_input, idx_array, idx_matrix):
        u = self.get_u()
        assert isinstance(idx_input, int) and idx_input <= len(u)
        black_box_perturbed = self.copy()
        u_perturbed = copy.deepcopy(u)
        if isinstance(u_perturbed[idx_input], np.ndarray):
            if type == "complex":
                u_perturbed[idx_input] = u_perturbed[idx_input].astype(complex)
            if u_perturbed[idx_input].ndim == 1:
                u_perturbed[idx_input][idx_array] = \
                    perturbe_scalar(u_perturbed[idx_input][idx_array], type, epsilon)
            else:
                assert u_perturbed[idx_input].ndim == 2
                u_perturbed[idx_input][idx_array][idx_matrix] = \
                    perturbe_scalar(u_perturbed[idx_input][idx_array][idx_matrix], type, epsilon)
        else:
            assert not hasattr(u_perturbed[idx_input], "__len__") # u[idx_input] is a scalar
            u_perturbed[idx_input] = perturbe_scalar(u_perturbed[idx_input], type, epsilon)
        black_box_perturbed.set_u(u_perturbed)
        return black_box_perturbed

    def validate_forward(self, idx_input, idx_array, idx_matrix) :
        # no forward validation with u containing sparse matrices
        """
        validates l = [0,..0], l[idx_input] = 1 in the forward mode
        idx_input: indicates the number of input scalars/vectors/matrices
        idx_array: if u[idx_input] is an array, we validate l[idx_input] = [0,..,0], l[idx_input][idx_array] = 1
        """
        # bumping
        epsilon = 10e-7
        black_box_plus = self.perturbe_black_box("plus", epsilon, idx_input, idx_array, idx_matrix)
        black_box_minus = self.perturbe_black_box("minus", epsilon, idx_input, idx_array, idx_matrix)
        diff_qoi_bumping = (black_box_plus.evaluate() - black_box_minus.evaluate()) / (2 * epsilon)

        # complex variable trick
        epsilon = 10e-20
        black_box_complex = self.perturbe_black_box("complex", epsilon, idx_input, idx_array, idx_matrix)
        diff_qoi_complex_trick = black_box_complex.evaluate(is_complex=True).imag / epsilon

        # forward mode
        diff_u = self.zero_u()
        if isinstance(diff_u[idx_input], np.ndarray):
            if diff_u[idx_input].ndim == 1:
                diff_u[idx_input][idx_array] = 1
            else:
                assert diff_u[idx_input].ndim == 2
                diff_u[idx_input][idx_array][idx_matrix] = 1
        else:
            diff_u[idx_input] = 1
        diff_qoi = self.forward(diff_u)

        if sparse.issparse(diff_qoi_bumping) :
            diff_qoi_bumping = diff_qoi_bumping.toarray()
        if sparse.issparse(diff_qoi_complex_trick) :
            diff_qoi_complex_trick = diff_qoi_complex_trick.toarray()
        if sparse.issparse(diff_qoi):
            diff_qoi = diff_qoi.toarray()

        err_complex = abs(diff_qoi_bumping - diff_qoi_complex_trick)
        err_forward = abs(diff_qoi_complex_trick - diff_qoi)

        # To Do: store the relative error bit in a new function
        if isinstance(err_complex, np.ndarray) and isinstance(err_forward, np.ndarray):
            rel_err_complex = np.zeros(err_complex.shape)
            rel_err_forward = np.zeros(err_forward.shape)
            if err_complex.ndim == 1 and err_forward.ndim == 1:
                for i in range(len(err_complex)) :
                    rel_err_complex[i] = err_complex[i] / 10e-16 if abs(diff_qoi_complex_trick[i]) < 10e-16 else \
                        err_complex[i] / diff_qoi_complex_trick[i]
                    rel_err_forward[i] = err_forward[i] / 10e-16 if abs(diff_qoi[i]) < 10e-16 else err_forward[i] / \
                                                                                                   diff_qoi[i]
            else:
                assert err_complex.ndim == 2 and err_forward.ndim == 2
                for i in range(err_complex.shape[0]) :
                    for j in range(err_complex.shape[1]) :
                        rel_err_complex[i][j] = err_complex[i][j] / 10e-16 if abs(diff_qoi_complex_trick[i][j]) < 10e-16 \
                            else err_complex[i][j] / diff_qoi_complex_trick[i][j]
                        rel_err_forward[i][j] = err_forward[i][j] / 10e-16 if abs(diff_qoi[i][j]) < 10e-16 else \
                            err_forward[i][j] / diff_qoi[i][j]

        else :
            rel_err_complex = err_complex / 10e-16 if abs(
                diff_qoi_complex_trick) < 10e-16 else err_complex / diff_qoi_complex_trick
            rel_err_forward = err_forward / 10e-16 if abs(diff_qoi) < 10e-16 else err_forward / diff_qoi

        print("Difference between the complex variable trick and bumping is: ", rel_err_complex)
        print("Difference of delta between the forward mode and complex variable trick is:", rel_err_forward)

    @abstractmethod
    def validate_reverse(self, diff_u, qoi_bar) :
        pass

    def validate(self, idx_input_forward_validation, diff_u, qoi_bar,
                 idx_array_forward_validation = np.nan,
                 idx_matrix_forward_validation = np.nan) :
        self.validate_forward(idx_input_forward_validation,
                              idx_array_forward_validation,
                              idx_matrix_forward_validation)
        self.validate_reverse(diff_u, qoi_bar)
