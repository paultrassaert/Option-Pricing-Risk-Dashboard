import pytest
import numpy as np
from models import black_scholes, calculate_greeks, calculate_implied_volatility, monte_carlo_simulation

# --- TEST 1: ACCURACY (KNOWN CASE) ---
def test_black_scholes_known_value():
    """
    Test against a known textbook value.
    S=100, K=100, T=1, r=0.05, sigma=0.2
    The Call price should be approximately 10.4506
    """
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.0, 0.2
    
    # Calculate Call price
    call_price = black_scholes(S, K, T, r, q, sigma, "call")
    
    # Verify with a tolerance of 0.001
    assert abs(call_price - 10.4506) < 0.001

# --- TEST 2: BOUNDARY LOGIC (NO NEGATIVE PRICES) ---
def test_no_negative_prices():
    """
    An option price can never be negative, even with extreme parameters.
    """
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.02, 0.5
    price = black_scholes(S, K, T, r, q, sigma, "call")
    assert price >= 0

# --- TEST 3: FINANCIAL LAW (PUT-CALL PARITY) ---
def test_put_call_parity():
    """
    Verifies the fundamental relationship: Call - Put = S - K * exp(-rT)
    This test proves that the core financial logic is respected.
    """
    S, K, T, r, q, sigma = 100, 110, 0.5, 0.05, 0.02, 0.25
    
    call_price = black_scholes(S, K, T, r, q, sigma, "call")
    put_price = black_scholes(S, K, T, r, q, sigma, "put")
    
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    
    # Verify that Left-Hand Side (LHS) equals Right-Hand Side (RHS)
    assert abs(lhs - rhs) < 0.0001

# --- TEST 4: GREEKS (CALL DELTA vs PUT DELTA) ---
def test_delta_relationship():
    """
    The difference between Call Delta and Put Delta must be equal to 1.
    Relationship: Delta(Call) - Delta(Put) = 1
    """
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.02, 0.2
    delta_c, _, _, _, _ = calculate_greeks(S, K, T, r, q, sigma, "call")
    delta_p, _, _, _, _ = calculate_greeks(S, K, T, r, q, sigma, "put")

    expected_diff = np.exp(-q * T)
    
    assert abs(delta_c - delta_p - expected_diff) < 0.0001


# --- TEST 5: ROUND TRIP (Volatility -> Price -> Volatility) ---
def test_implied_volatility_round_trip():
    """
    Verifies that the Newton-Raphson algorithm retrieves the original volatility.
    If we price an option with Sigma=30%, the IV should return 30%.
    """
    S, K, T, r, q = 100, 100, 1, 0.05, 0.02
    sigma_input = 0.30 # 30%
    
    # Calculate the "Target" market price
    market_price = black_scholes(S, K, T, r, q, sigma_input, "call")
    
    # Ask the model to find the implied volatility
    sigma_found = calculate_implied_volatility(market_price, S, K, T, r, q, "call")
    
    # Check for convergence (with a tiny error margin)
    assert abs(sigma_input - sigma_found) < 0.0001

# --- TEST 6: MONTE CARLO vs BLACK-SCHOLES (Convergence) ---
def test_monte_carlo_convergence():
    """
    With a high number of simulations, Monte Carlo should converge to the Black-Scholes price.
    Note: Due to stochastic nature, we allow a wider tolerance (e.g., +/- 2%).
    """
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.02, 0.2
    
    # Exact Price (Analytical Solution)
    bs_price = black_scholes(S, K, T, r, q, sigma, "call")
    
    # Estimated Price 
    np.random.seed(42) 
    mc_price, _ = monte_carlo_simulation(S, K, T, r, q, sigma, N=10000, option_type="call")
    
    # Check that the relative error is within 2%
    error_margin = abs(mc_price - bs_price) / bs_price
    assert error_margin < 0.02