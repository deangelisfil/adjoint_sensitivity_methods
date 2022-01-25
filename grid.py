# To do: add the grid notation throughout the project as I did for the bs_pde_standard folder
# and hereby, only have from parameters import grid
# To do: add S within the grid and include its functionalities forward and reverse
class Grid:
    def __init__(self, N, J, delta_S, delta_t):
        self.N = N
        self.J = J
        self.delta_S = delta_S
        self.delta_t = delta_t

