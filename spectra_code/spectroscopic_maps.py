
a0 = 0.5291772

class Spectroscopic_Maps:
    # Uses TIP4P as the default when not defined elsewhere
    def __init__(self):
        return None
    def w_map(self, E):
        w = 3760.2 - (3541.7 * E) - (152677 * (E ** 2))
        return w
    def mu_map(self, E):
        mu = 0.1646 + (11.39 * E) + (63.41 * (E ** 2))
        return mu
    def x_map(self, w):
        x = 0.19285 - (1.7261e-5 * w)
        return x
    def p_map(self, w):
        p = 1.6466 + (5.7692e-4 * w)
        return p
    def intra_map(self, Ei, Ej, xi, xj, pi, pj):
        intra = ((-1361 + (27165 * (Ei + Ej))) * xi * xj) - (1.887 * pi * pj)
        return intra

class TIP3P_Map(Spectroscopic_Maps):
    def w_map(self, E):
        w = 3742.81 - (4884.72 * E) - (65278.36 * (E**2))
        return w
    def mu_map(self, E):
        mu = 0.12 + (12.28 * E)
        return mu
    def x_map(self, w):
        x = (0.1019 - (9.0611e-6 * w)) / a0
        return x

class SPCE_Map(Spectroscopic_Maps):
    def w_map(self, E):
        w = 3762 - (5060 * E) - (86225 * (E**2))
        return w
    def mu_map(self, E):
        mu = 0.7112 + (75.58 * E)
        return mu
    def x_map(self, w):
        x = 0.1934 - (1.75e-5 * w)
        return x
    def p_map(self, w):
        p = 1.611 + (5.893e-4 * w)
        return p
    def intra_map(self, Ei, Ej, xi, xj, pi, pj):
        intra = ((-1789 + (23852 * (Ei + Ej))) * xi * xj) - (1.966 * pi * pj)
        return intra

class TIP4P_Map(Spectroscopic_Maps):
    def w_map(self, E):
        w = 3760.2 - (3541.7 * E) - (152677 * (E ** 2))
        return w
    def mu_map(self, E):
        mu = 0.1646 + (11.39 * E) + (63.41 * (E ** 2))
        return mu
    def x_map(self, w):
        x = 0.19285 - (1.7261e-5 * w)
        return x
    def p_map(self, w):
        p = 1.6466 + (5.7692e-4 * w)
        return p
    def intra_map(self, Ei, Ej, xi, xj, pi, pj):
        intra = ((-1361 + (27165 * (Ei + Ej))) * xi * xj) - (1.887 * pi * pj)
        return intra
