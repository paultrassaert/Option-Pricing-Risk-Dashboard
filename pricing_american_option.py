import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from scipy.stats import norm


# ---- BINOMIAL TREE (COX-ROSS-RUBINSTEIN) ----
def binomial_tree_american(S, K, T, r, q, sigma,  N=100, option_type="call"):
    """
    Price of an American option via Binomial Tree (Cox-Ross-Rubinstein Model).
    """
    if T <= 0: return max(0, (S - K) if option_type == "call" else (K - S))
    # Model parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt)) # Up factor
    d = 1 / u # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Security
    if not (0 < p < 1):
        p = max(0, min(1, p))

    # Initialization of asset prices at maturity
    asset_prices = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))

    # Initialization of option values at maturity
    if option_type.lower() == 'call':
        option_values = np.maximum(0, asset_prices - K)
    elif option_type.lower() == 'put':
        option_values = np.maximum(0, K - asset_prices)
    else : 
        raise ValueError("Option type must be 'call' or 'put'")
    
    # Backward Induction
    discount_factor = np.exp(-r * dt)
    for t in range(N - 1, -1, -1):
        # Value if held
        continuation_values = discount_factor * (p * option_values[:-1] + (1 - p) * option_values[1:])

        # Asset price at this step
        asset_prices = asset_prices[:-1] * d 

        # Value if exercised now 
        if option_type.lower() == 'call':
            intrinsic_values = np.maximum(0, asset_prices - K)
        elif option_type.lower() == 'put':
            intrinsic_values = np.maximum(0, K - asset_prices)
        else : 
            raise ValueError("Option type must be 'call' or 'put'")
        
        # The max of the two
        option_values = np.maximum(continuation_values, intrinsic_values)

    return option_values[0]

# ---- TRINOMIAL TREE ----
def trinomial_tree_american(S, K, T, r, q, sigma, N=500, option_type="call"):
    """
    Trinomial Tree.
    Price can go up (u), down (d), or stay flat (m).
    Converges faster and more 'smoothly' than the binomial model.
    """
    if T <= 0: return max(0, (S - K) if option_type == "call" else (K - S))
    # Model parameters
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = 1 / u
    m = 1.0 # Middle step

    # Probabilities
    sqrt_dt = np.sqrt(dt)
    drift = r - q - 0.5 * sigma**2
    pu = 0.5 * ((sigma**2 * dt + (drift * dt)**2) / dx**2 + (drift * dt) / dx)
    pd = 0.5 * ((sigma**2 * dt + (drift * dt)**2) / dx**2 - (drift * dt) / dx)
    pm = 1 - pu - pd

    if pu < 0 or pd < 0 or pm < 0:
        raise ValueError(
            f"Numerical instability: Negative probabilities detected (pu={pu:.4f}, pm={pm:.4f}, pd={pd:.4f}). "
            f"Increase the number of steps N (currently N={N}) to reduce dt."
        )
    discount_factor = np.exp(-r * dt)

    # Initialization of asset prices at maturity
    j = np.arange(N, -N - 1, -1)
    asset_prices = S * np.exp(j * dx)

    # Initialization payoff at maturity
    if option_type.lower() == 'call':
        values = np.maximum(0, asset_prices - K)
    elif option_type.lower() == 'put':
        values = np.maximum(0, K - asset_prices)
    else : 
        raise ValueError("Option type must be 'call' or 'put'")
    
    # Backward Induction
    for t in range(N -1, -1, -1):
        continuation_values = discount_factor * (pu * values[:-2] + pm * values[1:-1] + pd * values[2:])

        j_step = np.arange(t, -t - 1, -1)
        asset_prices_step = S * np.exp(j_step * dx)

        if option_type.lower() == 'call':
            intrinsic_values = np.maximum(0, asset_prices_step - K)
        elif option_type.lower() == 'put':
            intrinsic_values = np.maximum(0, K - asset_prices_step)
        else : 
            raise ValueError("Option type must be 'call' or 'put'")
        
        values = np.maximum(continuation_values, intrinsic_values)
    
    return values[0]

# ---- FINITE DIFFERENCE METHOD - THETA SCHEME ----

def fdm_theta_american(S_0, K, T, r, q, sigma, M=100, N=100, option_type="call", theta=0.5):
    """
    American Option Pricing via the Theta-Scheme (Generalization).
    
    PARAMETERS:
    -----------
    theta : float
        0.0 = Explicit (Unstable if dt is large)
        1.0 = Implicit (Stable but 1st order accuracy)
        0.5 = Crank-Nicolson (Stable and 2nd order accuracy) -> THE STANDARD
    """
    # Grid configuration
    S_max = K * np.exp(3 * sigma * np.sqrt(T))
    dS = S_max / M 
    dt = T / N
    S_values = np.linspace(0, S_max, M+1)
    grid = np.zeros((M+1,N+1))

    # Initial grid at Maturity
    if option_type.lower() == 'call':
        grid[:, N] = np.maximum(0, S_values - K)
    elif option_type.lower() == 'put':
        grid[:, N] = np.maximum(0, K - S_values)
    else : 
        raise ValueError("Option type must be 'call' or 'put'")
    
    i = np.arange(1,M)
    drift = (r - q) * i * dt
    diffusion = (sigma**2 * i**2 * dt)
    alpha = 0.5 * (diffusion - drift)
    beta  = 1.0 - diffusion - r * dt
    gamma = 0.5 * (diffusion + drift)

    # Matrix A (Left Side - Unknown Future - Implicit Part)
    A_lower = -theta * alpha
    A_diag  = 1 + theta * (diffusion + r*dt) # Sign and r*dt correction
    A_upper = -theta * gamma
    A_mat = sparse.diags([A_lower[1:], A_diag, A_upper[:-1]], offsets=[-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Matrix B (Right Side - Known Past - Explicit Part)
    B_lower = (1 - theta) * alpha
    B_diag  = 1 - (1 - theta) * (diffusion + r*dt)
    B_upper = (1 - theta) * gamma
    B_mat = sparse.diags([B_lower[1:], B_diag, B_upper[:-1]], offsets=[-1, 0, 1], shape=(M-1, M-1), format='csc')

    # Pre-calculate LU decomposition of A for fast solving at each step
    solve_A = splu(A_mat)

    payoff = grid[: N]
    for t in range(N-1, -1, -1):
        V_next = grid[1:M, t+1]

        # Right-hand side term: B * V_{t+1}
        RHS = B_mat.dot(V_next)
        
        # Boundary values at time n and t+1
        if option_type.lower() == 'call':
            bc_bottom_t = 0
            bc_top_t    = (S_max - K * np.exp(-r*(T - t*dt)))
        elif option_type.lower() == 'put':
            bc_bottom_t = K * np.exp(-r*(T - t*dt))
            bc_top_t    = 0
        else : 
            raise ValueError("Option type must be 'call' or 'put'")
        
        bc_bottom_tp1 = grid[0, t+1]
        bc_top_tp1    = grid[M, t+1]
        
        # Boundary terms enter the equation via alpha and gamma
        # We inject the known boundary values
        correction_bottom = (1-theta)*alpha[0]*bc_bottom_tp1 + theta*alpha[0]*bc_bottom_t
        correction_top    = (1-theta)*gamma[-1]*bc_top_tp1 + theta*gamma[-1]*bc_top_t
        
        RHS[0] += correction_bottom
        RHS[-1] += correction_top
        
        # LINEAR SYSTEM SOLVER: A * V_t = RHS
        V_t_interior = solve_A.solve(RHS)
        
        # Update grid
        grid[1:M, t] = V_t_interior
        grid[0, t]   = bc_bottom_t
        grid[M, t]   = bc_top_t
        
        # AMERICAN CONDITION (Check for Early Exercise)
        if option_type.lower() == 'call':
            intrinsic = np.maximum(S_values - K, 0)
        elif option_type.lower() =='put':
            intrinsic = np.maximum(K - S_values, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
            
        grid[:, t] = np.maximum(grid[:, t], intrinsic)

        # Final Interpolation
    f_interp = interp1d(S_values, grid[:, 0], kind='linear')
    try:
        price = f_interp(S_0)
    except ValueError:
        price = grid[np.abs(S_values - S_0).argmin(), 0]
        
    return float(price)

# ---- MONTE CARLO - LONGSTAFF-SCHWARTZ METHOD ----

def longstaff_schwartz_american(S, K, T, r, q, sigma, N=50, simulations=10000, option_type='call', seed=None):
    """
    American Option Pricing via Monte Carlo (Longstaff-Schwartz Method).
    Uses Least Squares Regression (LSM) to estimate the conditional expectation of continuation value.
    
    Parameters:
    -----------
    N : Number of time steps (e.g., 50 per year)
    simulations : Number of simulations (e.g., 10,000 paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    df = np.exp(-r * dt) # Discount factor for one time step
    Z = np.random.normal(0, 1, (simulations,N))
    S_paths = np.zeros((simulations, N+1))
    S_paths[:, 0]= S

    # Construct paths
    for t in range(1, N+1):
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

    # Cashflow Initialization at Maturity
    if option_type.lower() == 'call':
        payoff = np.maximum(S_paths[:, -1] - K, 0)
    elif option_type.lower() =='put':
        payoff = np.maximum(K - S_paths[:, -1], 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    cashflows = payoff

    # Backward
    for t in range(N-1, 0, -1):
        S_t = S_paths[:, t]

        #Intrinsic Value (Immediate Exercise)
        if option_type.lower() == 'call':
            intrinsic = np.maximum(S_t - K, 0)
        elif option_type.lower() =='put':
            intrinsic = np.maximum(K - S_t, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
    
        # Select "In The Money" (ITM) paths
        itm_mask = intrinsic > 0
        if np.any(itm_mask):
            X = S_t[itm_mask]

            # Discount future cash flows back to time step t
            Y = cashflows[itm_mask] * df
            
            # Scale X to prevent numerical overflow/instability
            X_std = X / K 
            coeffs = np.polynomial.laguerre.lagfit(X_std, Y, 3)
            continuation_val = np.polynomial.laguerre.lagval(X_std, coeffs)
            
            # Exercise decision
            exercise_mask = intrinsic[itm_mask] > continuation_val
            
            # Cash flow updates:
            # ITM paths where early exercise is optimal
            itm_indices = np.where(itm_mask)[0]
            exercise_indices = itm_indices[exercise_mask]
            cashflows[exercise_indices] = intrinsic[exercise_indices]
            
            # ITM paths not exercised AND OTM paths
            # We must discount the cash flows for those that are not exercised
            non_exercise_indices = np.setdiff1d(np.arange(simulations), exercise_indices)
            cashflows[non_exercise_indices] *= df
        else:
            # If no paths are ITM, simply discount all current cash flows
            cashflows *= df

    # Final price: average of discounted cash flows at time 0
    return np.mean(cashflows * df)

# ---- BJERKSUND-STENSLAND METHOD (2002) ----

def bjerksund_stensland_2002(S, K, T, r, q, sigma, option_type="call"):
    """
    Bjerksund-Stensland (2002) closed-form approximation formula.
    Industry standard for fast pricing of American options.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate
    q : float : Dividend yield
    sigma : float : Volatility
    option_type : str : "call" or "put"
    """
    
    # Option Type Handling (Put-Call Symmetry)
    # The Bjerksund-Stensland model is natively derived for Call options.
    # To price a Put, we use the property: Put(S, K, T, r, q) = Call(K, S, T, q, r)
    if option_type.lower() == 'put':
        return bjerksund_stensland_2002(K, S, T, q, r, sigma, option_type='call')
    
    if option_type.lower() != "call":
        raise ValueError("Option type must be 'call' or 'put'")

    # Cost of carry (b)
    b = r - q
    
    # Trivial Case (Early exercise never optimal)
    # If the cost of carry (b) >= risk-free rate (r), the American Call 
    # value is equal to the European Call (Black-Scholes).
    if b >= r:
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Exercise Boundary Parameters
    beta = (0.5 - b / sigma**2) + np.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / sigma**2)
    B_inf = (beta / (beta - 1)) * K
    B_0 = max(K, (r / (r - b)) * K)

    # Internal function to calculate the exercise boundary I(t)
    def get_I(t):
        # Using the stable form based on B_0 and B_inf
        h = -(b * t + 2 * sigma * np.sqrt(t)) * (B_0 / (B_inf - B_0))
        return B_0 + (B_inf - B_0) * (1 - np.exp(h))

    I1 = get_I(0.5 * T)
    I2 = get_I(T)

    # If the current price is already above the boundary, exercise immediately
    if S >= I2:
        return max(S - K, 0.0)

    # Auxiliary Function Phi
    def phi(S_in, T_in, gamma, H, I):
        kappa = (2 * b) / (sigma**2) + (2 * gamma - 1)
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2
        d = -(np.log(S_in / H) + (b + (gamma - 0.5) * sigma**2) * T_in) / (sigma * np.sqrt(T_in))
        
        # Black-Scholes term + Reflection term
        res = np.exp(lam * T_in) * (S_in**gamma) * (
            norm.cdf(d) - ((I / S_in)**kappa) * norm.cdf(d - (2 * np.log(I / S_in)) / (sigma * np.sqrt(T_in)))
        )
        return res

    alpha1 = (I1 - K) * (I1**-beta)
    alpha2 = (I2 - K) * (I2**-beta)

    # Final Calculation (12-term summation)
    v = (alpha2 * S**beta 
         - alpha2 * phi(S, T, beta, I2, I2) 
         + phi(S, T, 1, I2, I2) 
         - phi(S, T, 1, I1, I2) 
         - K * phi(S, T, 0, I2, I2) 
         + K * phi(S, T, 0, I1, I2) 
         + alpha1 * phi(S, T, beta, I1, I2) 
         - alpha1 * phi(S, T, beta, I1, I1) 
         + phi(S, T, 1, I1, I1) 
         - phi(S, T, 1, K, I1) 
         - K * phi(S, T, 0, I1, I1) 
         + K * phi(S, T, 0, K, I1))

    return max(v, S - K)

# ---- EXERCISE BOUNDARY CALCULATION ----

def calculate_exercise_boundary(S, K, T, r, q, sigma, option_type, steps=50):
    """
    Calculates the early exercise boundary for American options.
    Uses the Bjerksund-Stensland approximation logic and Put-Call Symmetry.
    """
    # Edge cases: Expired option or zero volatility
    if T <= 0 or sigma <= 0:
        return np.linspace(0, T, steps), [K] * steps

    # Setup Parameters based on Option Type 
    if option_type.lower() == 'put':
        # Swap risk-free rate and dividend yield for the calculation
        r_eff, q_eff = q, r
    else:
        # Standard Call parameters
        r_eff, q_eff = r, q

    # Cost of carry
    b_eff = r_eff - q_eff

    # Early exercise check: 
    # An American Call on a non-dividend paying stock (q <= 0) is never optimal to exercise early.
    if option_type.lower() == 'call' and q_eff <= 0:
        return np.linspace(0, T, steps), [np.nan] * steps

    # --- Bjerksund-Stensland Constants ---
    sigma2 = sigma**2
    phi = 0.5 - b_eff / sigma2
    
    # Calculate Beta
    radicand = phi**2 + 2 * r_eff / sigma2
    beta = phi + np.sqrt(max(0, radicand))

    # Mathematical safety check (Beta must be > 1 for the formula to hold)
    if beta <= 1:
        return np.linspace(0, T, steps), [np.nan] * steps

    # Calculate Asymptotic Boundaries
    B_inf = (beta / (beta - 1)) * K
    
    # B_0: Boundary at time 0 (expiration)
    if r_eff > b_eff:
        B_0 = max(K, (r_eff / (r_eff - b_eff)) * K)
    else:
        B_0 = K

    # --- Time Loop ---
    times = np.linspace(0.001, T, steps)
    boundary_prices = []

    for t in times:
        # Interpolation formula for the boundary I(t)
        h = -(b_eff * t + 2 * sigma * np.sqrt(t)) * (B_0 / (B_inf - B_0))
        i_t = B_0 + (B_inf - B_0) * (1 - np.exp(h))

        # --- Final Adjustment ---
        if option_type.lower() == 'put':
            # Apply Symmetry: The Put boundary is derived from the symmetric Call boundary
            if i_t != 0:
                boundary_prices.append(K**2 / i_t)
            else:
                boundary_prices.append(0)
        else:
            boundary_prices.append(i_t)

    return times, boundary_prices