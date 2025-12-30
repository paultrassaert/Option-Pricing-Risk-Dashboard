# ---- IMPORTS BIBLIOGRAPHY ---

import numpy as np
import scipy.stats
from scipy.stats import norm
import yfinance as yf
import pandas as pd

# ---- BLACK-SCHOLES ----

def black_scholes(S, K, T, r, sigma,  option_type = "call"):
    """
    Calculates the price of a European option using the Black-Scholes model.

    S: Current stock price
    K: Exercise price (Strike)
    T: Time to maturity (in years)
    r: Risk-free interest rate (e.g., 0.05 for 5%)
    sigma: Volatility (e.g., 0.2 for 20%)
    option_type: "call" or "put"
    """
    d1 = (np.log(S/K) + (r+sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else : 
        raise ValueError("Option type must be 'call' or 'put'")
    return price   

# ---- DATA RECOVERY ----     

def get_data(ticker):
    """
    Fetches the current price and calculates the 1-year historical volatility.
    """
    # Download 1 year of historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")

    #  Retrieve the latest closing price (Spot Price S)
    S = df['Close'].iloc[-1]

    #  Calculate Volatility (Sigma)
    # Calculate daily logarithmic returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate the standard deviation (std) of returns
    daily_volatility = df['Log_Returns'].std()

    # Annualize the volatility (multiply by square root of 252 trading days)
    sigma = daily_volatility * np.sqrt(252)

    return S, sigma

# ---- GREEKS ----

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates Delta, Gamma, Vega, Theta, and Rho.
    Optimized: Single conditional block for Call/Put logic.
    """
    # Common pre-calculations 
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Probability pre-calculations 
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_minus_d2 = norm.cdf(-d2)
    
    # GREEKS
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * sqrt_T * pdf_d1 * 0.01
    theta_term1 = -(S * sigma * pdf_d1) / (2 * sqrt_T)

    if option_type.lower() == "call":
        delta = cdf_d1
        theta = theta_term1 - r * K * np.exp(-r * T) * cdf_d2
        rho = K * T * np.exp(-r * T) * cdf_d2 * 0.01
        
    elif option_type.lower() =="put":      
        delta = cdf_d1 - 1
        theta = theta_term1 + r * K * np.exp(-r * T) * cdf_minus_d2
        rho = -K * T * np.exp(-r * T) * cdf_minus_d2 * 0.01
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    theta = theta / 365

    return delta, gamma, vega, theta, rho

# ---- MONTE-CARLO ----

def monte_carlo_simulation(S, K, T, r, sigma, N, option_type="call", time_steps=252, seed=None):
    """
    Simulates N price paths to visualize uncertainty using Geometric Brownian Motion (GBM).
    
    Returns:
    - mc_price: The estimated option price (Call)
    - S_paths: A NumPy array containing all simulated price trajectories
    """

    if seed is not None:
        np.random.seed(seed)
    # Define the time increment (step size)
    dt = T / time_steps

    # Initialize the price paths matrix (Rows: Time, Columns: Simulations)
    S_paths = np.zeros((time_steps + 1, N))
    S_paths[0] = S

    # Generate random shocks from a standard normal distribution
    # Using Z ~ N(0,1) for each step and simulation
    Z = np.random.standard_normal((time_steps,N))

    for t in range(1,time_steps + 1):
        # Update price based on risk-neutral drift and volatility shock
        S_paths[t] = S_paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt)* Z[t-1])

    # Calculate the Payoff at maturity (T)
    last_prices = S_paths[-1]
    
    if option_type.lower() == "call":
        # Call Payoff: max(S_T - K, 0)
        payoffs = np.maximum(last_prices - K, 0)
    elif option_type.lower() == "put":
        # Put Payoff: max(K - S_T, 0)
        payoffs = np.maximum(K - last_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount the expected payoff to present value
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    
    return mc_price, S_paths

# ---- VALUE-AT-RISK ----

def calculate_var(S_paths, K , T, r, option_type="call", confidence_level=0.95):
    """
    Calculates Value at Risk (VaR) at maturity based on Monte Carlo simulation.
    Returns the VaR value (loss amount) and the full P&L distribution.
    
    Note: This assumes a LONG position (Buyer of the option).
    """

    #Calculate Final Payoffs for each scenario
    final_prices = S_paths[-1]

    if option_type.lower() == "call":
        payoffs = np.maximum(final_prices - K, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - final_prices, 0)

   # Update (Present Value)
    simulated_option_prices = payoffs * np.exp(-r * T)
    
    # Calculate P&L (Profit & Loss) for each scenario
    price_mean = np.mean(simulated_option_prices)
    pnl_distribution = simulated_option_prices - price_mean

    #Calculate VaR
    percentile_cutoff = (1-confidence_level)*100
    var_value = np.percentile(pnl_distribution, percentile_cutoff)

    return var_value, pnl_distribution

# ---- IMPLIED VOLATILITY ----

def calculate_implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Calculates the Implied Volatility using the Newton-Raphson numerical method.
    We seek the sigma such that: BlackScholes(sigma) - MarketPrice = 0
    """
    MAX_ITERATIONS = 100
    PRECISION = 10e-5
    sigma = 0.5
    
    for i in range (MAX_ITERATIONS):
        # Calculate theorical price based on sigma
        price = black_scholes(S, K, T, r, sigma, option_type)

        # Calculate Vega
        d1 = (np.log(S / K)+( r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)

        diff = market_price - price
        if abs(diff) < PRECISION:
            return sigma
        
        #Safety Check
        if abs(vega) < 1.0e-8:
            break

        sigma = sigma + diff / vega

    return sigma
