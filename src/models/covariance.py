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
