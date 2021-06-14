from payoff import *
import numpy as np
from bs_pde.bs_pde import Bs_pde
from parameters import *
from bs_pde_adjoint.bs_pde_adjoint import Bs_pde_adjoint
from calibration_sensitivity.calibration_sensitivity import Calibration_sensitivity

if __name__ == "__main__":

    stock = Payoff(lambda S, is_complex : S,
                          lambda S, is_complex : np.ones(S.shape),
                          T)
    call = Call(K, T)
    put = Put(K, T)
    option_list = [stock, call, put]


    # example of decorator
    def loss(P_market: np.ndarray) -> callable :
        def loss_fn(P_model: np.ndarray) -> float :
            return 0.5 * sum((P_model - P_market) ** 2)
        return loss_fn


    def diff_loss(P_market: np.ndarray) -> callable :
        def diff_loss_fn(P_model: np.ndarray) -> float :
            return P_model - P_market
        return diff_loss_fn

    diff_u = [0,0,1]
    # diff_u = 3 * np.random.randn(3)
    qoi_bar = 1
    # qoi_bar = np.random.randn()


    BS_PDE = Bs_pde(S0, sigma, r, put)
    print(BS_PDE)
    BS_PDE.validate(0, diff_u, qoi_bar)
    BS_PDE_AMERICAN = Bs_pde(S0, sigma, r, put, american=True)
    print(BS_PDE_AMERICAN)
    BS_PDE_AMERICAN.validate(0, diff_u, qoi_bar)
    BS_PDE_ADJOINT = Bs_pde_adjoint(S0, sigma, r, put)
    print(BS_PDE_ADJOINT)
    BS_PDE_ADJOINT.validate(0, diff_u, qoi_bar)

    P_market = np.ones(len(option_list))
    loss_fn = loss(P_market)
    diff_loss_fn = diff_loss(P_market)

    CALIBRATION_SENSITIVITY = Calibration_sensitivity(S0, sigma, r, option_list, loss_fn, diff_loss_fn)
    print(CALIBRATION_SENSITIVITY.evaluate())
    print(CALIBRATION_SENSITIVITY.forward(diff_u))
    CALIBRATION_SENSITIVITY.validate(1, diff_u, qoi_bar)