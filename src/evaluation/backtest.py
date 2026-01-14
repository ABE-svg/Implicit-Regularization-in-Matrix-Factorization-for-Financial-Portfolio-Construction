import numpy as np
from .metrics import sharpe, volatility, max_drawdown
from ..models.portfolio import MinVariancePortfolio

def backtest(R_test: np.ndarray, Sigma: np.ndarray, ridge: float):
    port = MinVariancePortfolio(ridge).fit(Sigma)
    r = port.returns(R_test)

    # wealth + drawdown series (for plots)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak

    return {
        "sharpe": sharpe(r),
        "vol": volatility(r),
        "max_dd": max_drawdown(r),
        "weights": port.weights_,
        "returns": r,
        "wealth": wealth,
        "drawdown": dd
    }
