import numpy as np

class EntryMaskOperator:
    def __init__(self, n, omega_upper):
        self.n = n
        self.omega_upper = omega_upper

    @staticmethod
    def random(n, frac, seed, include_diag=True):
        rng = np.random.default_rng(seed)
        upper = [(i, j) for i in range(n) for j in range(i, n)]
        if not include_diag:
            upper = [(i, j) for (i, j) in upper if i != j]
        m = int(frac * len(upper))
        omega = [upper[k] for k in rng.choice(len(upper), m, replace=False)]
        return EntryMaskOperator(n, omega)

    def forward(self, X):
        return np.array([X[i, j] for (i, j) in self.omega_upper])

    def adjoint(self, r):
        G = np.zeros((self.n, self.n))
        for k, (i, j) in enumerate(self.omega_upper):
            if i == j:
                G[i, i] += r[k]
            else:
                G[i, j] += r[k]
                G[j, i] += r[k]
        return G
