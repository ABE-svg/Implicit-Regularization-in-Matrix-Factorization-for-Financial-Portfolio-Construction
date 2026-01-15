import numpy as np

def sharpe(x):
    return float(np.mean(x) / np.std(x, ddof=1))

def volatility(x):
    return float(np.std(x, ddof=1))

def max_drawdown(x):
    w = np.cumprod(1 + x)
    p = np.maximum.accumulate(w)
    return float(np.min((w - p) / p))
