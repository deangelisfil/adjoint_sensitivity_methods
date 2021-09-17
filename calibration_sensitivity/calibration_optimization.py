from calibration_sensitivity.calibration_sensitivity_adjoint.calibration_sensitivity_adjoint import Calibration_sensitivity_adjoint
import numpy as np


def calibration_optimization(calibration: Calibration_sensitivity_adjoint, nbr_epoch: int, lr: float):
    for epoch in range(nbr_epoch):
        # loss = calibration.evaluate() # TO DO: optimize it when doing forward and then backward
        sigma_bar, r_bar = calibration.reverse()
        # gradient descent step
        sigma_new = calibration.sigma - lr * sigma_bar
        r_new = calibration.r - lr * r_bar
        # update theta = {sigma, r}
        u_new = np.array([sigma_new, r_new])
        calibration.set_u(u_new)
        # print("Epoch:", epoch, " Loss:", np.round(loss, 10), " sigma_bar:", np.round(sigma_bar, 5), " r_bar:", np.round(r_bar, 5))





