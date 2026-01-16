import os
import numpy as np
import pandas as pd

from src.config import Config
from src.etl.utils import download_prices_yfinance, log_returns, rolling_windows

from src.operators.portfolio_variance import PortfolioVarianceOperator
from src.models.covariance import SampleCovariance, FactorizedGDPortfolioVarCovariance
from src.algorithms.optimize import FactorizedGD
from src.evaluation.backtest import backtest

from src.plots.charts import (
    plot_trace, plot_spectrum_X, plot_spectrum_U,
    plot_cumulative_wealth, plot_drawdown
)


def summarize(window_metrics, key):
    """Mean/std across rolling windows for a given metric key."""
    arr = np.asarray([m[key] for m in window_metrics], dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def main():
    cfg = Config()

    os.makedirs("assets/img", exist_ok=True)
    os.makedirs("assets/results", exist_ok=True)

    # 1) Download data and compute returns
    prices = download_prices_yfinance(cfg.tickers, cfg.start, cfg.end)
    print("Shape of prices matrix:", prices.shape)

    returns = log_returns(prices).dropna(how="any")  # ensure clean windows
    print("Shape of returns matrix:", returns.shape)

    results = {"sample": [], "tiny": [], "big": []}

    # Keep last window objects for diagnostic plots (trace/spectra)
    last_Sigma_sample = None
    last_est_tiny = None
    last_est_big = None

    k = 0
    for R_train, R_test in rolling_windows(returns, cfg.T_train, cfg.T_test, cfg.step):
        k += 1
        print(f"[window {k}] train={R_train.shape}, test={R_test.shape}")

        n = R_train.shape[1]

        # 2) Finance-native operator: portfolio variance measurements
        m = getattr(cfg, "m_portfolios", None)
        if m is None:
            m = 2 * n
        else:
            if m <= 0 or m > 20 * n:
                m = 2 * n

        A = PortfolioVarianceOperator.random(n=n, m=m, seed=cfg.seed, kind=cfg.w_kind)

        # 3) Sample covariance baseline
        Sigma_sample = SampleCovariance().fit(R_train.values).covariance_

        # 4) Factorized GD estimators (tiny vs big init)
        gd_tiny = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_tiny, cfg.seed, cfg.log_every)
        gd_big  = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_big,  cfg.seed, cfg.log_every)

        est_tiny = FactorizedGDPortfolioVarCovariance(A, gd_tiny).fit(R_train.values)
        est_big  = FactorizedGDPortfolioVarCovariance(A, gd_big).fit(R_train.values)

        # 5) Backtest on this window
        results["sample"].append(backtest(R_test.values, Sigma_sample, cfg.ridge))
        results["tiny"].append(backtest(R_test.values, est_tiny.covariance_, cfg.ridge))
        results["big"].append(backtest(R_test.values, est_big.covariance_, cfg.ridge))

        # Store last window estimators for diagnostics plots
        last_Sigma_sample = Sigma_sample
        last_est_tiny = est_tiny
        last_est_big = est_big

        # Optional safety: limit number of windows for very fast runs
        max_windows = getattr(cfg, "max_windows", None)
        if max_windows is not None and k >= max_windows:
            print(f"Reached max_windows={max_windows}. Stopping early.")
            break

    if k == 0:
        raise RuntimeError("No rolling windows were generated. Check T_train, T_test, step vs total T.")

    # 6) Save aggregated performance table (mean ± std across windows)
    rows = []
    for key, label in [
        ("sample", "Sample covariance"),
        ("tiny",   "Factorized GD (tiny init) [implicit reg]"),
        ("big",    "Factorized GD (big init)  [weak reg]"),
    ]:
        sh_m, sh_s = summarize(results[key], "sharpe")
        vol_m, vol_s = summarize(results[key], "vol")
        mdd_m, mdd_s = summarize(results[key], "max_dd")
        rows.append({
            "Method": label,
            "Sharpe_mean": sh_m, "Sharpe_std": sh_s,
            "Vol_mean": vol_m, "Vol_std": vol_s,
            "MaxDD_mean": mdd_m, "MaxDD_std": mdd_s,
        })

    perf_df = pd.DataFrame(rows)
    perf_path = "assets/results/performance_table.csv"
    perf_df.to_csv(perf_path, index=False)
    print("\nSaved:", perf_path)
    print(perf_df)

    # 7) Save weights table (last window weights)
    weights_df = pd.DataFrame({
        "ticker": returns.columns,
        "w_sample": results["sample"][-1]["weights"],
        "w_tiny": results["tiny"][-1]["weights"],
        "w_big": results["big"][-1]["weights"],
    })
    weights_path = "assets/results/weights_table.csv"
    weights_df.to_csv(weights_path, index=False)
    print("Saved:", weights_path)

    # 8) Save figures (use last window for wealth/drawdown; diagnostics from last estimators)
    plot_cumulative_wealth({
        "Sample": results["sample"][-1]["wealth"],
        "Tiny init": results["tiny"][-1]["wealth"],
        "Big init": results["big"][-1]["wealth"],
    }, filename="cumulative_wealth.png")

    plot_drawdown({
        "Sample": results["sample"][-1]["drawdown"],
        "Tiny init": results["tiny"][-1]["drawdown"],
        "Big init": results["big"][-1]["drawdown"],
    }, filename="drawdown.png")

    # Diagnostics plots: trace and spectra (last window)
    plot_trace(
        {"tiny init": last_est_tiny.result.history, "big init": last_est_big.result.history},
        filename="trace_tiny_vs_big.png"
    )

    plot_spectrum_X(
        {"sample": last_Sigma_sample, "tiny init": last_est_tiny.covariance_, "big init": last_est_big.covariance_},
        filename="spectrum_X_sample_tiny_big.png"
    )

    plot_spectrum_U(
        {"U tiny": last_est_tiny.result.U, "U big": last_est_big.result.U},
        filename="spectrum_U_tiny_big.png"
    )

    print("\nDone. Outputs written to assets/results and assets/img.")


if __name__ == "__main__":
    main()
