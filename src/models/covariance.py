import numpy as np
from dataclasses import dataclass
from ..etl.utils import sample_covariance

@dataclass
class FactorizedResult:
    U: object
    X: object
    history: object

class SampleCovariance:
    def fit(self, R):
        self.covariance_ = sample_covariance(R)
        return self

class FactorizedGDCovariance:
    def __init__(self, operator, solver):
        self.operator = operator
        self.solver = solver

    def fit(self, R):
        S = sample_covariance(R)
        y = self.operator.forward(S)
        U, X, hist = self.solver.fit(self.operator, y, S.shape[0])
        self.covariance_ = X
        self.result = FactorizedResult(U, X, hist)
        return self

class FactorizedGDPortfolioVarCovariance:
    def __init__(self, operator, solver):
        self.operator = operator
        self.solver = solver

    def fit(self, R):
        # target variances y_i = Var(R w_i)
        W = self.operator.W                  # (m,n)
        p = R @ W.T                          # (T,m) portfolio returns
        y = p.var(axis=0, ddof=1)            # (m,)

        U, X, hist = self.solver.fit(self.operator, y, W.shape[1])
        self.covariance_ = X
        self.result = FactorizedResult(U, X, hist)
        return self