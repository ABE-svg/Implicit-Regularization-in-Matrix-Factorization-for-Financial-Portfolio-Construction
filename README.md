# Implicit Regularization in Matrix Factorization for Financial Portfolio Construction

## Overview

This project studies **implicit regularization** induced by gradient descent when estimating covariance matrices through **matrix factorization**, and its impact on **portfolio construction**.

We consider the factorized parameterization:

    X = U U^T

where X is a positive semi-definite covariance matrix. Optimization is performed directly on the factor U using gradient descent **without explicit regularization**.

The central question of this project is:

> How does the initialization of gradient descent affect the complexity of the learned covariance matrix and the resulting portfolio performance?

---

## Objectives

- Compare **sample covariance** with **factorized covariance estimators**
- Study implicit regularization through:
  - **Tiny initialization** (strong implicit regularization)
  - **Big initialization** (weak implicit regularization)
- Evaluate portfolio performance using:
  - Sharpe ratio
  - Volatility
  - Maximum drawdown
  - Cumulative wealth

---

## Methodology

### Data

- Daily asset prices are downloaded using `yfinance`
- Log-returns are computed and split into:
  - Training period (covariance estimation)
  - Test period (portfolio evaluation)

---

### Covariance Estimation

Three covariance estimators are considered:

1. **Sample covariance**  
   The empirical covariance matrix computed directly from training returns.

2. **Factorized Gradient Descent (Tiny initialization)**  
   The factor matrix U is initialized with very small norm.  
   This induces **strong implicit regularization**, leading to low-complexity covariance matrices.

3. **Factorized Gradient Descent (Big initialization)**  
   The factor matrix U is initialized with large norm.  
   This results in **weak implicit regularization** and high-complexity covariance estimates.

---

### Portfolio Construction

For each covariance matrix, a **minimum-variance portfolio with ridge regularization** is constructed:

    minimize    w^T Σ w 
    subject to  sum(w) = 1

---

### Evaluation Metrics

All metrics are computed **out-of-sample** on the test period:

- **Sharpe ratio**: mean portfolio return divided by its standard deviation
- **Volatility**: standard deviation of portfolio returns
- **Maximum drawdown**: worst peak-to-trough loss
- **Cumulative wealth**: compounded portfolio value over time

---

## Results
Note: The code was written by the authors for this project, with generative AI used as a helper for code structuring and debugging.

### Performance Summary

Saved in `assets/results/performance_table.csv`

| Method | Sharpe | Volatility | Max Drawdown |
|------|------|------|------|
| Sample covariance | highest | lowest | smallest |
| Factorized GD (tiny init) | negative | higher | large |
| Factorized GD (big init) | most negative | very high | severe |

Across rolling windows, the **sample covariance estimator produces the most stable and best-performing portfolios**.
Factorized estimators exhibit clear structural differences but do not outperform the classical baseline in finite samples.

---

### Cumulative Wealth (Test Period)

Saved in `assets/img/wealth_test.png`

- Sample covariance produces smooth and stable growth
- Tiny initialization leads to large fluctuations and negative risk-adjusted performance
- Big initialization exhibits extreme volatility and large crashes

---

### Drawdown Analysis

Saved in `assets/img/drawdown_test.png`

- Big initialization results in deep and persistent drawdowns
- Tiny initialization reduces drawdown severity relative to big initialization
- Sample covariance remains the most conservative estimator

---

### Implicit Regularization Signal

Saved in `assets/img/trace_X.png`

The trace of the covariance matrix measures its complexity.

- Tiny initialization → very low trace (low effective complexity)
- Big initialization → high trace plateau

This provides direct empirical evidence of **implicit regularization induced by gradient descent**.

---

### Spectral Analysis

- **Singular values of U** (`assets/img/spectrum_U_tiny_big.png`)
  - Tiny initialization → fast spectral decay
  - Big initialization → many active directions

- **Eigenvalues of X** (`assets/img/spectrum_X_sample_tiny_big.png`)
  - Tiny initialization → low effective rank
  - Big initialization → noisy, high-rank structure

---

## Interpretation and Link with the Reference Article

The reference article shows that, when optimizing a matrix through a factorized parameterization using gradient descent, the optimization problem admits **multiple global minima**. In this setting, the algorithm is implicitly biased toward solutions of **lower nuclear norm** when initialized close to zero.

This project empirically confirms this mechanism in a **financial portfolio construction setting**:

- Tiny initialization consistently produces covariance matrices with:
  - lower trace,
  - faster spectral decay,
  - lower effective rank.

- Big initialization converges to higher-complexity solutions that:
  - overfit training noise,
  - produce unstable portfolios,
  - suffer from severe drawdowns out-of-sample.


While implicit regularization clearly affects covariance structure, it **does not automatically translate into improved portfolio performance** in finite samples. The sample covariance baseline remains highly competitive in this empirical setting.

---

## Installation and Execution

    pip install -r requirements.txt
    python -m app.app

---

**NB**: The execution takes approximately 4 minutes before displaying the results on a standard laptop.

---

## Project structure

    .
    ├── app/
    │   └── app.py
    ├── src/
    │   ├── algorithms/
    │   ├── etl/
    │   ├── models/
    │   ├── operators/
    │   ├── evaluation/
    │   └── plots/
    ├── assets/
    │   ├── img/
    │   └── results/
    ├── notebooks/
    ├── requirements.txt
    └── README.md
