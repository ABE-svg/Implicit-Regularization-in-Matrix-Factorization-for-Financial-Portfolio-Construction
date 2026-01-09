import numpy as np

class MinVariancePortfolio:
    def __init__(self, ridge):
        self.ridge = ridge

    def fit(self, Sigma):
        n = Sigma.shape[0]
        Sigma += self.ridge * np.eye(n)
        ones = np.ones(n)
        x = np.linalg.solve(Sigma, ones)
        self.weights_ = x / (ones @ x)
        return self

    def returns(self, R):
        return R @ self.weights_
