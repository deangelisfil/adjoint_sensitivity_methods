from auxiliary_functions import maximum
import numpy as np

class Function():
    def __init__(self, evaluate: callable, diff_evaluate: callable):
        self._evaluate = evaluate
        self._diff_evaluate = diff_evaluate

    def evaluate(self, S: np.ndarray, is_complex : bool = False) -> np.ndarray:
        return self._evaluate(S, is_complex)

    def diff_evaluate(self, S: np.ndarray, is_complex : bool = False):
        return self._diff_evaluate(S, is_complex)

class Call(Function):
    def __init__(self, K: float):
        self._evaluate = lambda S, is_complex: maximum(S - K, 0, is_complex)
        self._diff_evaluate = lambda S, is_complex: np.heaviside(S - K, 1)
        self.K = K

    def __repr__(self):
        return "Call with strike " + str(np.round(self.K, 5))

class Put(Function):
    def __init__(self, K: float):
        self._evaluate = lambda S, is_complex: maximum(K-S,0, is_complex)
        self._diff_evaluate = lambda S, is_complex: -np.heaviside(K-S, 1)
        self.K = K

    def __repr__(self):
        return "Put with strike " + str(np.round(self.K, 5))