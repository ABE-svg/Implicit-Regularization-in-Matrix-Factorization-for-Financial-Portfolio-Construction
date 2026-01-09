from .metrics import sharpe, volatility, max_drawdown
from ..models.portfolio import MinVariancePortfolio

def backtest(R_test, Sigma, ridge):
    port = MinVariancePortfolio(ridge).fit(Sigma)
    r = port.returns(R_test)
    return {
        "sharpe": sharpe(r),
        "vol": volatility(r),
        "max_dd": max_drawdown(r),
        "weights": port.weights_
    }
