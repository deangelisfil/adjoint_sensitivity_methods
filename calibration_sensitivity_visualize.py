from function import *
from parameters import *
from calibration_sensitivity.calibration_loss import Squared_error
from calibration_sensitivity.calibration_sensitivity_standard.calibration_sensitivity_standard \
    import Calibration_sensitivity_standard
from calibration_sensitivity_local_volatility import Calibration_sensitivity_local_volatility
from matplotlib import pyplot as plt
import time


def calibration_visualize(calibration_standard,
                          calibration_adjoint,
                          validate = False,
                          idx_input_forward_validation = 0,
                          diff_u = [1, 0],
                          qoi_bar =  1,
                          idx_array_forward_validation = np.nan):
    # TO DO: write the function in order for it to make sense
    lr = 0.1
    nbr_epochs = 10
    nbr_K = [2 ** k for k in range(5)]
    n = len(P_call_market)
    mid_idx = n // 2
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

        calibration_standard.option_list = option_list
        calibration_standard.loss = loss
        calibration_adjoint.option_list = option_list
        calibration_adjoint.loss = loss

        # standard calibration
        start_time = time.time()
        if validate:
            print("Standard Calibration:")
            calibration_standard.validate(idx_input_forward_validation, diff_u, qoi_bar, idx_array_forward_validation)
        calibration_standard.optimize(nbr_epochs, lr)
        end_time = time.time()
        time_standard_all.append(end_time - start_time)

        # adjoint calibration
        start_time = time.time()
        if validate:
            print("Adjoint Calibration:")
            calibration_adjoint.validate(idx_input_forward_validation, diff_u, qoi_bar, idx_array_forward_validation)
        calibration_adjoint.optimize(nbr_epochs, lr)
        end_time = time.time()
        time_adjoint_all.append(end_time - start_time)
    time_standard_all = np.array(time_standard_all) / time_standard_all[0]
    time_adjoint_all = np.array(time_adjoint_all) / time_adjoint_all[0]

    plt.plot(nbr_K, time_standard_all, linestyle='--', marker='*', color='r')
    plt.plot(nbr_K, time_adjoint_all, linestyle='--', marker='o', color='b')
    plt.legend(["Standard calibration: Reverse mode", "Adjoint calibration: Reverse mode"])
    plt.xlabel('Number of strikes $K$')
    plt.ylabel("Relative cost")
    if isinstance(calibration_standard, Calibration_sensitivity_standard):
        plt.title("Comparison of the execution times for the computation \n"
              "of calibration sensitivities in the Black-Scholes model")
    elif isinstance(calibration_standard, Calibration_sensitivity_local_volatility):
        plt.title("Comparison of the execution times for the computation \n"
              "of calibration sensitivities in the Local-Volatility model")
    else:
        raise TypeError("calibration object is neither BS nor LV calibration")
    plt.show()



