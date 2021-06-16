from function import *
from parameters import *
from calibration_sensitivity.calibration_sensitivity_adjoint.calibration_sensitivity_adjoint \
    import Calibration_sensitivity_adjoint
from calibration_sensitivity.calibration_loss import Squared_error
from calibration_sensitivity.calibration_optimization import calibration_optimization
from calibration_sensitivity.calibration_sensitivity_standard.calibration_sensitivity_standard \
    import Calibration_sensitivity_standard
from matplotlib import pyplot as plt
import time

if __name__ == "__main__":

    stock = Function(lambda S, is_complex : S,
                     lambda S, is_complex : np.ones(S.shape))
    call = Call(K)
    put = Put(K)
    option_list = [stock, call, put]

    diff_u = [0,0,1]
    # diff_u = 3 * np.random.randn(3)
    qoi_bar = 1
    # qoi_bar = np.random.randn()


    # BS_PDE = Bs_pde(S0, sigma, r, stock)
    # print(BS_PDE)
    # print("forward:", BS_PDE.forward(diff_u))
    # print("reverse:", BS_PDE.reverse(qoi_bar))
    # BS_PDE.validate(0, diff_u, qoi_bar)
    # BS_PDE_AMERICAN = Bs_pde(S0, sigma, r, stock, american=True)
    # print(BS_PDE_AMERICAN)
    # BS_PDE_AMERICAN.validate(0, diff_u, qoi_bar)
    # BS_PDE_ADJOINT = Bs_pde_adjoint(S0, sigma, r, stock)
    # print(BS_PDE_ADJOINT)
    # BS_PDE_ADJOINT.validate(0, diff_u, qoi_bar)

    # plt.plot(K_call_all, P_call_market)
    # plt.show()

    lr = 0.1
    nbr_epochs = 10
    n = len(P_call_market)
    mid_idx = n//2
    option_list = [Call(k) for k in K_call_all]
    nbr_K = [2**k for k in range(9)]
    # nbr_K = [10*k for k in range(n//10)]
    time_standard_all = []
    time_adjoint_all = []
    for K in nbr_K:
        if K == 1 or K == 0:
            P_market = P_call_market[mid_idx]
            strikes = [K_call_all[mid_idx]]
        else:
            lower_idx = mid_idx-K//2
            upper_idx = mid_idx+K//2
            P_market = P_call_market[ lower_idx : upper_idx]
            strikes = list(K_call_all[lower_idx : upper_idx])

        option_list = [Call(k) for k in strikes]
        loss = Squared_error(P_market)
        calibration_standard = Calibration_sensitivity_standard(S0, sigma, r, option_list, loss, american=False)
        calibration_adjoint = Calibration_sensitivity_adjoint(S0, sigma, r, option_list, loss)

        # standard calibration
        start_time = time.time()
        calibration_standard.validate(1, diff_u, qoi_bar)
        calibration_optimization(calibration_standard, nbr_epochs, lr)
        end_time = time.time()
        time_standard_all.append(end_time - start_time)

        # adjoint calibration
        start_time = time.time()
        # calibration_adjoint.validate(1, diff_u, qoi_bar)
        calibration_optimization(calibration_adjoint, nbr_epochs, lr)
        end_time = time.time()
        time_adjoint_all.append(end_time-start_time)

    plt.plot(nbr_K, time_standard_all)
    plt.plot(nbr_K, time_adjoint_all)
    plt.legend(["Standard calibration", "Adjoint calibration"])
    plt.show()




