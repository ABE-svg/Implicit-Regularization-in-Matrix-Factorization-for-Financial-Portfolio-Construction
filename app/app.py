from src.config import Config
from src.etl.utils import download_prices_yfinance, log_returns, train_test_split
from src.operators.entry_mask import EntryMaskOperator
from src.algorithms.optimize import FactorizedGD
from src.models.covariance import SampleCovariance, FactorizedGDCovariance
from src.evaluation.backtest import backtest
from src.plots.charts import plot_trace, plot_spectrum_X, plot_spectrum_U

def main():
    cfg = Config()

    prices = download_prices_yfinance(cfg.tickers, cfg.start, cfg.end)
    returns = log_returns(prices)
    R_train, R_test = train_test_split(returns, cfg.test_size)

    A = EntryMaskOperator.random(len(cfg.tickers), cfg.mask_frac, cfg.seed)

    Sigma_sample = SampleCovariance().fit(R_train.values).covariance_

    gd_tiny = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_tiny, cfg.seed, cfg.log_every)
    gd_big = FactorizedGD(cfg.lr, cfg.n_steps, cfg.init_scale_big, cfg.seed, cfg.log_every)

    est_tiny = FactorizedGDCovariance(A, gd_tiny).fit(R_train.values)
    est_big = FactorizedGDCovariance(A, gd_big).fit(R_train.values)

    print(backtest(R_test.values, Sigma_sample, cfg.ridge))
    print(backtest(R_test.values, est_tiny.covariance_, cfg.ridge))
    print(backtest(R_test.values, est_big.covariance_, cfg.ridge))

    plot_trace({"tiny": est_tiny.result.history, "big": est_big.result.history})
    plot_spectrum_X({"sample": Sigma_sample, "tiny": est_tiny.covariance_, "big": est_big.covariance_})
    plot_spectrum_U({"tiny": est_tiny.result.U, "big": est_big.result.U})

if __name__ == "__main__":
    main()
