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

def calibration_visualize(validate = False, diff_u = [1,0,0], qoi_bar =  1):
    # TO DO: write the function in order for it to make sense
    lr = 0.1
    nbr_epochs = 10
    n = len(P_call_market)
    mid_idx = n // 2
    nbr_K = [2 ** k for k in range(5)]
    # nbr_K = [10*k for k in range(n//10)]
    time_standard_all = []
    time_adjoint_all = []
    for K in nbr_K :
        if K == 1 or K == 0 :
            P_market = P_call_market[mid_idx]
            strikes = [K_call_all[mid_idx]]
        else :
            lower_idx = mid_idx - K // 2
            upper_idx = mid_idx + K // 2
            P_market = P_call_market[lower_idx : upper_idx]
            strikes = list(K_call_all[lower_idx : upper_idx])

        option_list = [Call(k) for k in strikes]
        loss = Squared_error(P_market)
        calibration_standard = Calibration_sensitivity_standard(S0, sigma, r, option_list, loss, american=False)
        calibration_adjoint = Calibration_sensitivity_adjoint(S0, sigma, r, option_list, loss)

        # standard calibration
        start_time = time.time()
        if validate:
            print("Standard Calibration:")
            calibration_standard.validate(1, diff_u, qoi_bar)
        calibration_optimization(calibration_standard, nbr_epochs, lr)
        end_time = time.time()
        time_standard_all.append(end_time - start_time)

        # adjoint calibration
        start_time = time.time()
        if validate:
            print("Adjoint Calibration:")
            calibration_adjoint.validate(1, diff_u, qoi_bar)
        calibration_optimization(calibration_adjoint, nbr_epochs, lr)
        end_time = time.time()
        time_adjoint_all.append(end_time - start_time)
    time_standard_all = np.array(time_standard_all) / time_standard_all[0]
    time_adjoint_all = np.array(time_adjoint_all) / time_adjoint_all[0]

    plt.plot(nbr_K, time_standard_all, linestyle='--', marker='*', color='r')
    plt.plot(nbr_K, time_adjoint_all, linestyle='--', marker='o', color='b')
    plt.legend(["Standard calibration: Reverse mode", "Adjoint calibration: Reverse mode"])
    plt.xlabel('Number of strikes $K$')
    plt.ylabel("Relative cost")
    plt.title("Comparison of the execution times for the computation \n"
              "of calibration sensitivities in the Black-Scholes model")
    plt.show()



