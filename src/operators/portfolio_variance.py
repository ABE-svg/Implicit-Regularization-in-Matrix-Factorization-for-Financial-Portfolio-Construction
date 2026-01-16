import numpy as np

class PortfolioVarianceOperator:
    """
    A(X)_i = w_i^T X w_i, for i=1..m
    Adjoint: A^*(r) = sum_i r_i * (w_i w_i^T)
    """
    def __init__(self, W: np.ndarray):
        # W shape: (m, n)
        self.W = W
        self.m, self.n = W.shape

    @staticmethod
    def random(n: int, m: int, seed: int, kind: str = "l2"):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((m, n))
        if kind == "l2":
            W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        elif kind == "simplex":
            W = np.abs(W)
            W /= (W.sum(axis=1, keepdims=True) + 1e-12)
        return PortfolioVarianceOperator(W)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # returns vector length m: diag(W X W^T)
        return np.einsum("bi,ij,bj->b", self.W, X, self.W)

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        # sum_i r_i w_i w_i^T
        return (self.W.T * r) @ self.W
