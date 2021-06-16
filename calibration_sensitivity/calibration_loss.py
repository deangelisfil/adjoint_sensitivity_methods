import numpy as np
from function import Function


class Calibration_loss() :
    def __init__(self, P_market: np.ndarray, loss_fn: Function) :
        """loss_fn is a function taking P_model and P_market (in this order) as an input"""
        self.P_market = P_market
        self.loss_fn = loss_fn

    def evaluate(self, P_model) :
        return self.loss_fn.evaluate(P_model, self.P_market)

    def diff_evaluate(self, P_model) :
        return self.loss_fn.diff_evaluate(P_model, self.P_market)

    def set_P_market(self, P_market) :
        self.P_market = P_market


class Squared_error(Calibration_loss) :
    def __init__(self, P_market: np.ndarray) :
        self.P_market = P_market
        self.loss_fn = Function(lambda P_model, P_market: 0.5 * sum((P_model - P_market) ** 2),
                                lambda P_model, P_market: P_model - P_market)