from function import *
from parameters import *
from calibration_sensitivity.calibration_loss import Squared_error
from calibration_sensitivity.calibration_sensitivity_standard.calibration_sensitivity_standard \
    import Calibration_sensitivity_standard
from calibration_sensitivity.calibration_sensitivity_adjoint.calibration_sensitivity_adjoint \
    import Calibration_sensitivity_adjoint
from splines.Cubic_splines_construction import Cubic_splines_construction
from splines.Cubic_splines_evaluation import Cubic_splines_evaluation
from B_construction.B_construction_time_invariant_ import B_construction_time_invariant
from bs_pde.bs_pde_standard.bs_pde_standard import Bs_pde_standard
from bs_pde.bs_pde_adjoint.bs_pde_adjoint import Bs_pde_adjoint



if __name__ == "__main__":

    stock = Function(lambda S, is_complex : S,
                     lambda S, is_complex : np.ones(S.shape))
    call = Call(K)
    put = Put(K)

    #diff_u = [0,1,0]
    diff_u = np.random.randn(3)
    #qoi_bar = 1
    qoi_bar = np.random.randn()

    S = np.array([S0 + j*delta_S for j in range(-J, J+1)])
    diff_S = diff_u[0] * np.ones(2*J + 1)
    S_p = np.array([0, 0.5, 1, 1.5, 2])
    sigma_p = np.array([1, 0.4, 0.2, 0.4, 1])
    #S = np.array([0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15])
    #diff_S = diff_u[0] * np.ones(len(S))
    #sigma_bar = np.ones(len(S))
    sigma_bar = np.random.randn(len(S))
    diff_sigma_p = np.array([1,0,0,0,0])
    #diff_sigma_p = np.random.randn(5)
    sigma_p_double_prime_bar = np.ones(S_p.shape)
    #sigma_p_double_prime_bar = np.random.randn(5)

    print("Cubic Spline construction:")
    cubic_splines_construction = Cubic_splines_construction(S_p, sigma_p)
    cubic_splines_construction.validate(0, diff_sigma_p, sigma_p_double_prime_bar)
    print("------------------")

    cubic_splines_evaluation = Cubic_splines_evaluation(S, cubic_splines_construction)
    print("Test cubic interpolation: ")
    print(cubic_splines_evaluation.evaluate())
    # print(cubic_splines_evaluation.forward([diff_S, diff_sigma_p]))
    # S_bar, sigma_p_bar = cubic_splines_evaluation.reverse(sigma_bar)
    # print(S_bar)
    # print(sigma_p_bar)
    cubic_splines_evaluation.validate_reverse([diff_S, diff_sigma_p], sigma_bar)
    print("---------------")

    print("B construction")
    sigma_v = cubic_splines_evaluation.evaluate()
    # sigma_v = np.array([sigma + abs(j) * delta_S//10 for j in range(-J, J+1)])
    diff_sigma_v = np.ones(len(S))
    B_bar = np.ones((2*J+1, 2*J+1))

    B = B_construction_time_invariant(S, sigma_v, r, delta_t, delta_S)
    B.validate_reverse([diff_S, diff_sigma_v, diff_u[2]], B_bar)
    print("---------------")

    BS_PDE = Bs_pde_standard(S0, sigma, r, stock)
    print(BS_PDE)
    BS_PDE.validate(0, diff_u, qoi_bar)
    BS_PDE_AMERICAN = Bs_pde_standard(S0, sigma, r, put, american=True)
    print(BS_PDE_AMERICAN)
    BS_PDE_AMERICAN.validate(0, diff_u, qoi_bar)
    BS_PDE_ADJOINT = Bs_pde_adjoint(S0, sigma, r, stock)
    print(BS_PDE_ADJOINT)
    BS_PDE_ADJOINT.validate(0, diff_u, qoi_bar)
    print("--------------")
    # plt.plot(K_call_all, P_call_market)
    # plt.show()

    n = len(P_call_market)
    mid_idx = n // 2
    K = 4
    lower_idx = mid_idx - K // 2
    upper_idx = mid_idx + K // 2
    P_market = P_call_market[lower_idx : upper_idx]
    strikes = list(K_call_all[lower_idx : upper_idx])

    option_list = [Call(k) for k in strikes]
    loss = Squared_error(P_market)
    calibration_standard = Calibration_sensitivity_standard(S0, sigma, r, option_list, loss, american=False)
    calibration_adjoint = Calibration_sensitivity_adjoint(S0, sigma, r, option_list, loss)

    # print("Standard Calibration:")
    # print(calibration_standard.evaluate())
    # print(calibration_standard.forward([diff_u[1], diff_u[2]]))
    # print(calibration_standard.reverse(qoi_bar))
    # calibration_standard.validate(0, [diff_u[1], diff_u[2]], qoi_bar)

    # print("Adjoint Calibration:")
    # print(calibration_adjoint.evaluate())
    # print(calibration_adjoint.forward([diff_u[1], diff_u[2]]))
    # print(calibration_adjoint.reverse(qoi_bar))
    # calibration_adjoint.validate(0, [diff_u[1], diff_u[2]], qoi_bar)

    # calibration_optimization(calibration_adjoint, 100, 0.1)

    #calibration_visualize(validate=True, diff_u=[diff_u[1], diff_u[2]], qoi_bar = qoi_bar)




