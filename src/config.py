from dataclasses import dataclass

@dataclass
class Config:
    tickers = ("SPY","QQQ","IWM","TLT","GLD","AAPL","MSFT","AMZN","JPM","XOM")
    start = "2019-01-01"
    end = "2024-01-01"
    test_size = 252

    mask_frac = 0.30
    include_diag = True
    seed = 42

    lr = 1e-3
    n_steps = 40_000
    log_every = 2000

    init_scale_tiny = 1e-4
    init_scale_big = 1e-1

    ridge = 1e-6
