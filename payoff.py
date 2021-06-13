#from abc import ABC, abstractmethod
from auxiliary_functions import maximum
import numpy as np

class Payoff():
    #@abstractmethod
    def __init__(self, payoff: callable, diff_payoff: callable, T):
        self._payoff = payoff
        self._diff_payoff = diff_payoff
        self.T = T

    def payoff(self, S: np.ndarray, is_complex : bool = False) -> np.ndarray:
        return self._payoff(S, is_complex)

    def diff_payoff(self, S: np.ndarray, is_complex : bool = False):
        return self._diff_payoff(S, is_complex)

class Call(Payoff):
    def __init__(self, K: float, T: float):
        self._payoff = lambda S, is_complex: maximum(S-K,0, is_complex)
        self._diff_payoff = lambda S, is_complex: np.heaviside(S-K, 1)
        self.K = K
        self.T = T

class Put(Payoff):
    def __init__(self, K: float, T: float):
        self._payoff = lambda S, is_complex: maximum(K-S,0, is_complex)
        self._diff_payoff = lambda S, is_complex: -np.heaviside(K-S, 1)
        self.K = K
        self.T = T


