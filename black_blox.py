from abc import ABC, abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity
import numpy as np

class Black_box(ABC):

    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def set_u(self, u: list):
        pass

    @abstractmethod
    def get_u(self) -> list:
        pass

    @abstractmethod
    def validate_forward(self, idx):
        pass

    @abstractmethod
    def validate_reverse(self, diff_u, qoi_bar):
        pass

    def validate(self, idx_forward_validation, diff_u, qoi_bar):
        self.validate_forward(idx_forward_validation)
        self.validate_reverse(diff_u, qoi_bar)
