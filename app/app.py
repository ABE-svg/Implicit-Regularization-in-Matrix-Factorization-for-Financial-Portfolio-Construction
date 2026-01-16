import os
from matplotlib.image import resample
import pandas as pd

from src.etl.utils import rolling_windows
from src.config import Config
from src.etl.utils import download_prices_yfinance, log_returns, train_test_split
from src.operators.entry_mask import EntryMaskOperator
from src.operators.portfolio_variance import PortfolioVarianceOperator
from src.models.covariance import SampleCovariance, FactorizedGDPortfolioVarCovariance
from src.algorithms.optimize import FactorizedGD
from src.models.covariance import SampleCovariance, FactorizedGDCovariance
from src.evaluation.backtest import backtest
from src.plots.charts import (
    plot_trace, plot_spectrum_X, plot_spectrum_U,
    plot_cumulative_wealth, plot_drawdown
)

def main():
    cfg = Config()

    os.makedirs("assets/img", exist_ok=True)
    os.makedirs("assets/results", exist_ok=True)

    # 1) Data
    prices = download_prices_yfinance(cfg.tickers, cfg.start, cfg.end)
    # Print the shape of the prices matrix
    print("Shape of prices matrix:", prices.shape)
    returns = log_returns(prices)
    results = {"sample": [], "tiny": [], "big": []}

    for R_train, R_test in rolling_windows(returns, cfg.T_train, cfg.T_test, cfg.step):
        n = len(cfg.tickers)
        m = 5 * n   # or 10*n
        A = PortfolioVarianceOperator.random(n=n, m=m, seed=cfg.seed, kind=cfg.w_kind)
        # A = EntryMaskOperator.random(len(cfg.tickers), cfg.mask_frac, cfg.seed, include_diag=cfg.include_diag)

        Sigma_sample = SampleCovariance().fit(R_train.values).covariance_

        gd_tiny = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_tiny, cfg.seed, cfg.log_every)
        gd_big  = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_big,  cfg.seed, cfg.log_every)
        est_tiny = FactorizedGDPortfolioVarCovariance(A, gd_tiny).fit(R_train.values)
        est_big  = FactorizedGDPortfolioVarCovariance(A, gd_big).fit(R_train.values)
        # est_tiny = FactorizedGDCovariance(A, gd_tiny).fit(R_train.values)
        # est_big  = FactorizedGDCovariance(A, gd_big).fit(R_train.values)

        results["sample"].append(backtest(R_test.values, Sigma_sample, cfg.ridge))
        results["tiny"].append(backtest(R_test.values, est_tiny.covariance_, cfg.ridge))
        results["big"].append(backtest(R_test.values, est_big.covariance_, cfg.ridge))
        # 5) Save results table
    table = pd.DataFrame([
        {"Method": "Sample covariance", "Sharpe": resample["sharpe"], "Vol": resample["vol"], "MaxDD": resample["max_dd"]},
        {"Method": "Factorized GD (tiny init)  [implicit reg]", "Sharpe": est_tiny["sharpe"], "Vol": est_tiny["vol"], "MaxDD": est_tiny["max_dd"]},
        {"Method": "Factorized GD (big init)   [weak reg]", "Sharpe": est_big["sharpe"], "Vol": est_big["vol"], "MaxDD": est_big["max_dd"]},
    ])

    table_path = "assets/results/performance_table.csv"
    table.to_csv(table_path, index=False)
    print("\nSaved:", table_path)
    print(table)

    # 6) Save weights table
    weights_df = pd.DataFrame({
        "ticker": cfg.tickers,
        "w_sample": resample["weights"],
        "w_tiny": est_tiny["weights"],
        "w_big": est_big["weights"],
    })
    weights_path = "assets/results/weights_table.csv"
    weights_df.to_csv(weights_path, index=False)
    print("Saved:", weights_path)

    # 7) Save figures (in assets/img/)
    plot_trace({"tiny init": est_tiny.result.history, "big init": est_big.result.history},
               filename="trace_tiny_vs_big.png")

    plot_spectrum_X({"sample": Sigma_sample, "tiny init": est_tiny.covariance_, "big init": est_big.covariance_},
                    filename="spectrum_X_sample_tiny_big.png")

    plot_spectrum_U({"U tiny": est_tiny.result.U, "U big": est_big.result.U},
                    filename="spectrum_U_tiny_big.png")

    plot_cumulative_wealth({
        "Sample": resample["wealth"],
        "Tiny init": est_tiny["wealth"],
        "Big init": est_big["wealth"],
    }, filename="cumulative_wealth.png")

    plot_drawdown({
        "Sample": resample["drawdown"],
        "Tiny init": est_tiny["drawdown"],
        "Big init": est_big["drawdown"],
    }, filename="drawdown.png")

if __name__ == "__main__":
    main()
