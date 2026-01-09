import numpy as np
from dataclasses import dataclass
from .optim_utils import trace_psd

@dataclass
class GDHistory:
    steps: list
    loss: list
    residual: list
    trace: list

class FactorizedGD:
    def __init__(self, lr, n_steps, init_scale, seed, log_every):
        self.lr = lr
        self.n_steps = n_steps
        self.init_scale = init_scale
        self.seed = seed
        self.log_every = log_every

    def fit(self, A, y, n):
        rng = np.random.default_rng(self.seed)
        U = self.init_scale * rng.standard_normal((n, n))
        hist = GDHistory([], [], [], [])

        for t in range(1, self.n_steps + 1):
            X = U @ U.T
            r = A.forward(X) - y
            loss = float(r @ r)

            if t == 1 or t % self.log_every == 0:
                hist.steps.append(t)
                hist.loss.append(loss)
                hist.residual.append(float(np.linalg.norm(r)))
                hist.trace.append(trace_psd(X))

            gradU = 4 * (A.adjoint(r) @ U)
            U -= self.lr * gradU

        return U, U @ U.T, hist
