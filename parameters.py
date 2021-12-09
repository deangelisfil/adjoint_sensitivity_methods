import numpy as np
import pandas as pd
from Grid import Grid
import os
# grid parameters
S0 = 1
sigma = 0.2
r = 0.1
K = 1
T = 1
N = 1024 # number of time steps
J = 32 # 2*J+1 is number of spatial steps
delta_S = S0/J
delta_t = T/N
assert(delta_t <= delta_S**2), "No stability of the finite difference scheme"
S = np.array([S0 + j * delta_S for j in range(-J, J + 1)]) + 0.01  # to avoid jumps in the binary search
grid = Grid(N, J, delta_S, delta_t)

spot = 4235.24
T = 1 # 1 week
cwd = os.getcwd()
data_path = os.path.join(cwd, "S&P500_options_21_06_14_w.csv")
df = pd.read_csv(data_path, thousands=',', sep=",")
df_call = df[df["Type"]=="Call"]
df_put = df[df["Type"]=="Put"]
K_call_all = df_call["Strike"].to_numpy() / spot
P_call_market = df_call["Midpoint"].to_numpy() / spot

