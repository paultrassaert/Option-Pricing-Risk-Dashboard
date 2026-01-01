import pytest
import numpy as np
from pricing_american_option import (
    binomial_tree_american, 
    trinomial_tree_american, 
    bjerksund_stensland_2002, 
    fdm_theta_american, 
    longstaff_schwartz_american
)

# --- FIXTURES (Common Parameters) ---
@pytest.fixture
def market_params():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.2
    }

# --- 1. MATURITY CONSISTENCY TESTS (T=0) ---
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_payoff_at_maturity(market_params, option_type):
    """Verifies that the price at T=0 equals the payoff exactly."""
    S, K = market_params["S"], market_params["K"]
    # T=0
    
    price = binomial_tree_american(S, K, 0, 0.05, 0.02, 0.2, option_type=option_type)
    expected = max(0, S - K) if option_type == "call" else max(0, K - S)
    
    assert price == pytest.approx(expected, abs=1e-5)

# --- 2. NO-DIVIDEND CALL TEST (No Early Exercise) ---
def test_american_call_no_div(market_params):
    """
    Property: An American Call on a non-dividend paying stock (q=0) 
    must be equal to the European price (Black-Scholes).
    Here we compare Bjerksund-Stensland vs Binomial as proxies.
    """
    params = market_params.copy()
    params["q"] = 0.0
    
    # Compare Bjerksund-Stensland to Binomial Tree
    price_bj = bjerksund_stensland_2002(params["S"], params["K"], params["T"], params["r"], 0.0, params["sigma"], "call")
    price_bin = binomial_tree_american(params["S"], params["K"], params["T"], params["r"], 0.0, params["sigma"], N=200, option_type="call")
    
    # They should converge to the same value within a small tolerance
    assert price_bj == pytest.approx(price_bin, rel=1e-2)

# --- 3. INTRINSIC VALUE TEST (Deep ITM) ---
def test_deep_itm_put():
    """A Deep In-The-Money Put must be >= Intrinsic Value."""
    S, K = 10.0, 100.0 # Deep ITM
    price = bjerksund_stensland_2002(S, K, 1.0, 0.05, 0.01, 0.2, "put")
    
    # Early exercise is optimal here, so price should be exactly K - S
    assert price >= (K - S)

# --- 4. NUMERICAL MODEL COMPARISON ---
def test_numerical_convergence(market_params):
    """Verifies that Trees and PDE models yield similar results."""
    S, K, T, r, q, sigma = market_params.values()
    
    p_bin = binomial_tree_american(S, K, T, r, q, sigma, N=200, option_type="put")
    p_tri = trinomial_tree_american(S, K, T, r, q, sigma, N=200, option_type="put")
    p_pde = fdm_theta_american(S, K, T, r, q, sigma, M=100, N=200, option_type="put")
    
    # Maximum tolerated deviation between numerical methods: 1%
    assert p_bin == pytest.approx(p_tri, rel=1e-2)
    assert p_tri == pytest.approx(p_pde, rel=1e-2)

# --- 5. MONTE CARLO TEST (LONGSTAFF-SCHWARTZ) ---
def test_longstaff_schwartz_accuracy(market_params):
    """Verifies that MC results are consistent with analytical models."""
    np.random.seed(42) 
    
    S, K, T, r, q, sigma = market_params.values()
    sigma = market_params.get('sigma', 0.2) 
    r = market_params.get('r', 0.05)

    ref_price = bjerksund_stensland_2002(S, K, T, r, q, sigma, "put")
    
    mc_price = longstaff_schwartz_american(
        S, K, T, r, q, sigma, 
        N=252, 
        simulations=50000, 
        option_type="put"
    )

    assert mc_price == pytest.approx(ref_price, rel=7e-2)