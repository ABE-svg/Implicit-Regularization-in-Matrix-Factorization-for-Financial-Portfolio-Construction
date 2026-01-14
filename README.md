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

    minimize    w^T Σ w + λ ||w||^2
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

### Performance Summary

Saved in `assets/results/performance_table.csv`

| Method | Sharpe | Volatility | Max Drawdown |
|------|------|------|------|
| Sample covariance | 0.3149 | 0.0006 | -0.4% |
| Factorized GD (tiny init) | 0.0568 | 0.0058 | -9.6% |
| Factorized GD (big init) | 0.0613 | 0.0162 | -25.3% |

Tiny initialization leads to **more stable portfolios and substantially lower drawdowns**.

---

### Cumulative Wealth (Test Period)

Saved in `assets/img/wealth_test.png`

- Sample covariance produces smooth but conservative growth
- Tiny initialization yields stable growth with controlled downside risk
- Big initialization exhibits high volatility and large crashes

---

### Drawdown Analysis

Saved in `assets/img/drawdown_test.png`

- Big initialization leads to drawdowns exceeding **25%**
- Tiny initialization significantly reduces downside risk
- Sample covariance remains stable but conservative

---

### Implicit Regularization Signal

Saved in `assets/img/trace_X.png`

The trace of the covariance matrix measures its complexity.

- Tiny initialization → very low trace (low nuclear norm)
- Big initialization → high trace plateau

This provides direct evidence of **implicit regularization induced by gradient descent**.

---

### Spectral Analysis

- **Singular values of U** (`assets/img/spectrum_U.png`)
  - Tiny initialization → fast decay
  - Big initialization → many active directions

- **Eigenvalues of X** (`assets/img/spectrum_X.png`)
  - Tiny initialization → low-rank covariance structure
  - Big initialization → noisy, high-rank covariance

---
## Installation and execution 

pip install -r requirements.txt
python -m app.app
---

## Project Structure

```text
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
