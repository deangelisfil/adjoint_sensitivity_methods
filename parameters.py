import payoff
import numpy as np

# grid parameters
S0 = 1
sigma = 0.2
r = 0.1
K = 1
T = 1
N = 16 # number of time steps
J = 4 # 2*J+1 is number of spatial steps
delta_S = S0/J
delta_t = T/N
assert(delta_t <= delta_S**2), "No stability of the finite difference scheme"

stock = payoff.Payoff(lambda S, is_complex : S,
               lambda S, is_complex : np.ones(S.shape),
               T)
call = payoff.Call(K, T)
put = payoff.Put(K, T)

payoff = put.payoff
diff_payoff = put.diff_payoff
