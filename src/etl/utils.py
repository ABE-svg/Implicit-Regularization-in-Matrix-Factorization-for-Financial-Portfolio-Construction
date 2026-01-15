import numpy as np
import pandas as pd

def download_prices_yfinance(tickers, start, end):
    import yfinance as yf
    prices = yf.download(list(tickers), start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    return prices.dropna()

def log_returns(prices: pd.DataFrame):
    return np.log(prices / prices.shift(1)).dropna()

def train_test_split(returns: pd.DataFrame, test_size: int):
    return returns.iloc[:-test_size], returns.iloc[-test_size:]

def sample_covariance(R: np.ndarray):
    Rc = R - R.mean(axis=0, keepdims=True)
    return (Rc.T @ Rc) / (R.shape[0] - 1)
