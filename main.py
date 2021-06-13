from payoff import *
import numpy as np
from bs_pde.bs_pde import Bs_pde
from parameters import *
from bs_pde_adjoint.bs_pde_adjoint import Bs_pde_adjoint

if __name__ == "__main__":

    # diff_u = [1,0,0]
    diff_u = 3 * np.random.randn(3)
    # qoi_bar = 1
    qoi_bar = np.random.randn()

    BS_PDE = Bs_pde(S0, sigma, r)
    print(BS_PDE)
    BS_PDE.validate(diff_u, qoi_bar)
    BS_PDE_AMERICAN = Bs_pde(S0, sigma, r, american=True)
    print(BS_PDE_AMERICAN)
    BS_PDE_AMERICAN.validate(diff_u, qoi_bar)
    BS_PDE_ADJOINT = Bs_pde_adjoint(S0, sigma, r)
    print(BS_PDE_ADJOINT)
    BS_PDE_ADJOINT.validate(diff_u, qoi_bar)
