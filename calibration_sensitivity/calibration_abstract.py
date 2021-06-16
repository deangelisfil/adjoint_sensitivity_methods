from abc import abstractmethod
from bs_pde.bs_pde_abstract import Bs_pde_abstract

class Calibration_sensitivity_abstract(Bs_pde_abstract):

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar=1):
        pass
