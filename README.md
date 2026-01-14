# Implicit Regularization in Matrix Factorization for Financial Portfolio Construction

## Overview

This project investigates **implicit regularization** induced by gradient descent when estimating covariance matrices through **matrix factorization**, and its impact on **portfolio construction**.

We consider the factorized parameterization:

\[
X = U U^\top
\]

where \(X\) is a positive semi-definite covariance matrix and optimization is performed over the factor \(U\) using gradient descent **without explicit regularization**.

The key question is:

> How does the initialization of gradient descent affect the complexity of the learned covariance matrix and the resulting portfolio performance?

---

## Objectives

- Compare **sample covariance** with **factorized covariance estimators**
- Study implicit regularization via:
  - **Tiny initialization** (strong implicit regularization)
  - **Big initialization** (weak implicit regularization)
- Evaluate economic performance using:
  - Sharpe ratio
  - Volatility
  - Maximum drawdown
  - Cumulative wealth

---

## Methodology

### Data
- Daily asset prices downloaded using `yfinance`
- Log-returns computed and split into:
  - Training period (covariance estimation)
  - Test period (portfolio evaluation)

### Covariance Estimation

We compare three covariance estimators:

1. **Sample covariance**
2. **Factorized GD (tiny init)**  
   Small initialization of \(U\), inducing strong implicit regularization and low nuclear norm solutions
3. **Factorized GD (big init)**  
   Large initialization of \(U\), leading to weak implicit regularization and higher-complexity solutions

---

### Portfolio Construction

For each covariance matrix, we construct a **minimum-variance portfolio with ridge regularization**:

\[
\min_w \; w^\top \Sigma w + \lambda \|w\|^2
\quad \text{s.t. } \mathbf{1}^\top w = 1
\]

---

### Evaluation Metrics

All metrics are computed **out-of-sample** on the test period:

- Sharpe ratio
- Volatility
- Maximum drawdown
- Cumulative wealth

---

## Results

### Performance Summary

Saved in `assets/results/performance_table.csv`

| Method | Sharpe | Volatility | Max Drawdown |
|------|------|------|------|
| Sample covariance | 0.3149 | 0.0006 | -0.4% |
| Factorized GD (tiny init) | 0.0568 | 0.0058 | -9.6% |
| Factorized GD (big init) | 0.0613 | 0.0162 | -25.3% |

Tiny initialization leads to **significantly lower drawdowns and more stable portfolios**.

---

### Cumulative Wealth

Saved in `assets/img/wealth_test.png`

- Sample covariance produces smooth but conservative growth
- Tiny initialization yields stable growth with controlled downside risk
- Big initialization leads to high volatility and large crashes

---

### Drawdown Analysis

Saved in `assets/img/drawdown_test.png`

- Big initialization exhibits drawdowns exceeding **25%**
- Tiny initialization substantially reduces downside risk
- Sample covariance remains stable but conservative

---

### Implicit Regularization Signal

Saved in `assets/img/trace_X.png`

The trace of the covariance matrix equals its nuclear norm for PSD matrices:

\[
\text{trace}(X) = \|X\|_*
\]

- Tiny initialization → very low nuclear norm
- Big initialization → high nuclear norm plateau

This provides direct evidence of **implicit regularization induced by gradient descent**.

---

### Spectral Analysis

- **Singular values of \(U\)** (`assets/img/spectrum_U.png`)
  - Tiny init → fast decay
  - Big init → many active directions

- **Eigenvalues of \(X\)** (`assets/img/spectrum_X.png`)
  - Tiny init → low-rank covariance structure
  - Big init → noisy, high-rank covariance

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
