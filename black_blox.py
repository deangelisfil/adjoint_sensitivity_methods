from abc import ABC, abstractmethod
from auxiliary_functions import check_forward_reverse_mode_identity

class Black_box(ABC):

    @abstractmethod
    def forward(self, diff_u):
        pass

    @abstractmethod
    def reverse(self, qoi_bar=1):
        pass

    @abstractmethod
    def evaluate(self, is_complex=False):
        pass

    @abstractmethod
    def validate(self, diff_u, qoi_bar):
        pass