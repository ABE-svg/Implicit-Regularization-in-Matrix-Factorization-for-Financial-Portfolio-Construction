import os
import pandas as pd

from src.config import Config
from src.etl.utils import download_prices_yfinance, log_returns, train_test_split
from src.operators.entry_mask import EntryMaskOperator
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
    returns = log_returns(prices)
    R_train, R_test = train_test_split(returns, cfg.test_size)

    # 2) Operator A(X)
    A = EntryMaskOperator.random(len(cfg.tickers), cfg.mask_frac, cfg.seed)

    # 3) Covariance estimates
    Sigma_sample = SampleCovariance().fit(R_train.values).covariance_

    gd_tiny = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_tiny, cfg.seed, cfg.log_every)
    gd_big  = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_big,  cfg.seed, cfg.log_every)

    est_tiny = FactorizedGDCovariance(A, gd_tiny).fit(R_train.values)
    est_big  = FactorizedGDCovariance(A, gd_big).fit(R_train.values)

    # 4) Backtests
    res_sample = backtest(R_test.values, Sigma_sample, cfg.ridge)
    res_tiny   = backtest(R_test.values, est_tiny.covariance_, cfg.ridge)
    res_big    = backtest(R_test.values, est_big.covariance_,  cfg.ridge)

    # 5) Save results table
    table = pd.DataFrame([
        {"Method": "Sample covariance", "Sharpe": res_sample["sharpe"], "Vol": res_sample["vol"], "MaxDD": res_sample["max_dd"]},
        {"Method": "Factorized GD (tiny init)  [implicit reg]", "Sharpe": res_tiny["sharpe"], "Vol": res_tiny["vol"], "MaxDD": res_tiny["max_dd"]},
        {"Method": "Factorized GD (big init)   [weak reg]", "Sharpe": res_big["sharpe"], "Vol": res_big["vol"], "MaxDD": res_big["max_dd"]},
    ])

    table_path = "assets/results/performance_table.csv"
    table.to_csv(table_path, index=False)
    print("\nSaved:", table_path)
    print(table)

    # 6) Save weights table
    weights_df = pd.DataFrame({
        "ticker": cfg.tickers,
        "w_sample": res_sample["weights"],
        "w_tiny": res_tiny["weights"],
        "w_big": res_big["weights"],
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
        "Sample": res_sample["wealth"],
        "Tiny init": res_tiny["wealth"],
        "Big init": res_big["wealth"],
    }, filename="cumulative_wealth.png")

    plot_drawdown({
        "Sample": res_sample["drawdown"],
        "Tiny init": res_tiny["drawdown"],
        "Big init": res_big["drawdown"],
    }, filename="drawdown.png")

if __name__ == "__main__":
    main()
