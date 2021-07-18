from function import *
from parameters import *
from bs_pde.bs_pde_standard.bs_pde_standard import Bs_pde_standard
from bs_pde.bs_pde_adjoint.bs_pde_adjoint import Bs_pde_adjoint
from calibration_sensitivity.calibration_visualize import calibration_visualize
from B_construction.B_construction_time_invariant_ import B_construction_time_invariant


if __name__ == "__main__":

    stock = Function(lambda S, is_complex : S,
                     lambda S, is_complex : np.ones(S.shape))
    call = Call(K)
    put = Put(K)

    diff_u = [1,0,0]
    # diff_u = 3 * np.random.randn(3)
    qoi_bar = 1
    # qoi_bar = np.random.randn()

    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_S = diff_u[0] * np.ones(2*J + 1)
    B_bar = np.ndarray((2*J+1, 2*J+1))

    B = B_construction_time_invariant(S, sigma, r, delta_t, delta_S)
    # B.validate_reverse([diff_S, diff_u[1], diff_u[2]], B_bar)


    BS_PDE = Bs_pde_standard(S0, sigma, r, stock)
    print(BS_PDE)
    BS_PDE.validate(0, diff_u, qoi_bar)
    BS_PDE_AMERICAN = Bs_pde_standard(S0, sigma, r, put, american=True)
    print(BS_PDE_AMERICAN)
    BS_PDE_AMERICAN.validate(0, diff_u, qoi_bar)
    BS_PDE_ADJOINT = Bs_pde_adjoint(S0, sigma, r, stock)
    print(BS_PDE_ADJOINT)
    BS_PDE_ADJOINT.validate(0, diff_u, qoi_bar)

    #plt.plot(K_call_all, P_call_market)
    #plt.show()

    # calibration_visualize(validate=True, diff_u=diff_u, qoi_bar = qoi_bar)




