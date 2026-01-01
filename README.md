# ⚡ Option Pricing & Risk Management Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
> **Interactive Quantitative Finance tool bridging theoretical pricing models with practical risk management.**

---

## Interface Preview

![Overview](assets/overview.png)

---

## Overview

This project is a comprehensive interactive dashboard built for traders and quantitative analysts. It leverages real-time market data to price European Options, analyze complex risk sensitivities (Greeks), and simulate future market scenarios using stochastic methods.

It solves a common problem: How to visualize the non-linear risks of an option portfolio?
By combining the analytical precision of Black-Scholes with the flexibility of Monte Carlo simulations, this tool provides a 360° view of market exposure.

## Key Features

### Pricing & Volatility
* **Real-Time Data Feed:** Automatic fetching of spot prices and historical volatility via `yfinance`.
* **Implied Volatility Solver:** Reverse-engineering of market volatility using the Newton-Raphson algorithm to match theoretical prices with market quotes.
* **Black-Scholes Engine:** Instant calculation of Call/Put premiums.

### Risk Analysis (The Greeks)
* **Dynamic Calculations:** Real-time computation of $\Delta$ (Delta), $\Gamma$ (Gamma), $\nu$ (Vega), $\Theta$ (Theta), and $\rho$ (Rho).
* **3D Sensitivity Surfaces:** Interactive heatmaps visualizing how Greeks evolve regarding Spot Price and Time to Maturity.

### Monte Carlo Simulations
* **Stochastic Engine:** Simulates thousands of price paths using Geometric Brownian Motion (GBM).
* **Convergence Analysis:** Compares numerical simulation results with analytical closed-form solutions.
* **Visual P&L:** Histogram distribution of potential payoffs at maturity.

### Risk Management
* **Value at Risk (VaR):** Statistical estimation of maximum potential loss (95% & 99% confidence intervals).
* **Delta-Hedging Strategy:** Automated calculation of the underlying shares required to immunize the portfolio against small price moves.

---

## Mathematical Framework

The dashboard relies on three core financial pillars:

### 1. Black-Scholes-Merton Model (Analytical Pricing)
<div align="center">
  <img src="https://math.vercel.app/?from=\color{black}\Large%20C(S,t)=N(d_1)S_t-N(d_2)Ke^{-r(T-t)}" alt="Black Scholes Formula" />
</div>

<br>

### 2. Geometric Brownian Motion (Stochastic Process)
<div align="center">
  <img src="https://math.vercel.app/?from=\color{black}\Large%20dS_t=\mu{S_t}dt+\sigma{S_t}dW_t" alt="GBM Formula" />
</div>

<br>

### 3. Newton-Raphson (Implied Volatility)
<div align="center">
  <img src="https://math.vercel.app/?from=\color{black}\Large%20\sigma_{n+1}=\sigma_n-\frac{C_{model}(\sigma_n)-C_{market}}{\mathcal{V}(\sigma_n)}" alt="Newton Raphson Formula" />
</div>
---

## Installation & Setup

**Prerequisites:** Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Option-Pricing-Risk-Dashboard.git
   cd Option-Pricing-Risk-Dashboard
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
   *The app will open in your browser at `http://localhost:8501`*

---

## Project Structure

```bash
Option-Pricing-Risk-Dashboard/
├── dashboard.py        # Entry point: Streamlit UI & Page layout
├── models.py           # Core Logic: BS, Greeks, Monte Carlo, IV algorithms
├── test_models.py      # Unit Tests: Pytest suite for math verification
├── requirements.txt    # Dependencies list
├── .gitignore          # Files to ignore (git configuration)
└── README.md           # Documentation
```

---

## Testing

Reliability is paramount in quantitative finance. This project includes a test suite using Pytest.

**Tests cover:**
* Theoretical accuracy (vs Textbook examples).
* Put-Call Parity checks (Arbitrage-free conditions).
* Implied Volatility "Round-Trip" (Input $\sigma$ $\to$ Price $\to$ Solver $\to$ Output $\sigma$).

To run the tests:
```bash
pytest test_models.py
```

---

## Disclaimer

*This project is for educational and research purposes only. It should not be used as the sole basis for real-money trading decisions. Options trading involves significant risk.*

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

**Author:** Paul Trassaert - *Mathematical & Computer Science Student*
