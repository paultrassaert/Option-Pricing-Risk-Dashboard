import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from models import get_data, black_scholes, calculate_greeks, monte_carlo_simulation, calculate_var, calculate_implied_volatility
from pricing_american_option import (binomial_tree_american, trinomial_tree_american, bjerksund_stensland_2002,
                                      longstaff_schwartz_american, fdm_theta_american, calculate_exercise_boundary)

@st.cache_data(ttl=600) 
def get_data_cached(ticker_symbol):
    return get_data(ticker_symbol)

@st.cache_data
def binomial_cached(S, K, T, r, q, sigma, N, opt_type):
    return binomial_tree_american(S, K, T, r, q, sigma, N, opt_type)

@st.cache_data
def trinomial_cached(S, K, T, r, q, sigma, N, opt_type):
    return trinomial_tree_american(S, K, T, r, q, sigma, N, opt_type)

@st.cache_data
def pde_cached(S, K, T, r, q, sigma, M, N, opt_type):
    return fdm_theta_american(S, K, T, r, q, sigma, M, N, opt_type)

@st.cache_data
def mc_amer_cached(S, K, T, r, q, sigma, N_steps, N_sims, opt_type):
    return longstaff_schwartz_american(S, K, T, r, q, sigma, N=N_steps, simulations=N_sims, option_type=opt_type)

@st.cache_data
def calculate_greeks_surface(current_price, K, T, r, q, sigma, option_type):
    vol_spot = np.linspace(current_price * 0.8, current_price * 1.2, 15)
    vol_time = np.linspace(0.02, T, 15) 
    X, Y = np.meshgrid(vol_spot, vol_time)
    d_grid, g_grid, v_grid, t_grid, r_grid = calculate_greeks(X, K, Y, r, q, sigma, option_type)
    return vol_spot, vol_time, d_grid, g_grid, v_grid, t_grid, r_grid

@st.cache_data
def calculate_pnl_grid(S_range, K, T, r, q, sigma, opt_type, am_bj_price, position_size):
    values = []
    for s in S_range:
        val = bjerksund_stensland_2002(s, K, T, r, q, sigma, opt_type)
        values.append((val - am_bj_price) * position_size * 100)
    return np.array(values)


# --- CONFIGURATION ---
st.set_page_config(page_title="Option Pricer and Risk", layout="wide", page_icon="⚡")
st.title("⚡ Option Pricing and Risk Dashboard")
st.markdown("Multi-Model American Valuation • Greeks Risk Profile • Stochastic Simulations • Portfolio Risk & Hedging")

# --- STYLE ---
st.markdown("""
<style>
    .stApp {
        background-color: #e6e9ef;
    }
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #0f293a;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #0f293a;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1a425e;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-darkgrid')

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Market Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="KO")

# --- DATA LOADING ---
try:
    current_price, hist_vol, div_yield = get_data_cached(ticker)
    st.sidebar.success(f"Data loaded for {ticker}")
except Exception as e:
    st.error(f"Technical Error: {e}")
    st.stop()

# SESSION STATE MANAGEMENT (STRIKE PRICE)
    
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ticker
    st.session_state.strike_k = float(current_price)
    st.session_state.vol_val = float(hist_vol * 100)

# Detect if Ticker changed since last run
if st.session_state.last_ticker != ticker:
    st.session_state.strike_k = float(current_price) 
    st.session_state.vol_val = float(hist_vol * 100)
    st.session_state.last_ticker = ticker

def update_strike():
    st.session_state.strike_k = st.session_state.input_strike_widget

# --- SIDEBAR INPUTS ---
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
T = st.sidebar.number_input("Time to Expiration (Years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.045, format="%.3f")
q = st.sidebar.number_input("Dividend Yield (q)", value=float(div_yield), format="%.3f", help="Yield automatically fetched from Yahoo Finance but editable.")
n_steps = st.sidebar.number_input("Time Steps (Discretization)", value=200, step =1, help="Number of time steps for the simulation (e.g., 252 for daily steps over 1 year).")
use_implied_vol = st.sidebar.checkbox("Calibrate with Market Price (IV)")
if use_implied_vol:
    default_price = black_scholes(current_price, float(st.session_state.strike_k), T, r, q, 0.4, option_type)

    if 'iv_price_key' not in st.session_state:
        st.session_state.iv_price_key = float(default_price)
    
    market_price_input = st.sidebar.number_input(
        "Market Option Price ($)",
        key='iv_price_key', 
        step=0.1,
        format="%.2f"
    )
    
    K_val = float(st.session_state.strike_k)
    if option_type == "Call":
        intrinsic_val = max(0, current_price - K_val)
    else:
        intrinsic_val = max(0, K_val - current_price)
        
    if market_price_input <= intrinsic_val:
        st.sidebar.error("⚠️ Price Impossible!")
        st.sidebar.caption(f"Min value is ${intrinsic_val:.2f} (Intrinsic)")
        sigma = 0.20 
    else:
        try:
            implied_vol = calculate_implied_volatility(market_price_input, current_price, K_val, T, r, q, option_type)
            sigma = implied_vol
            st.sidebar.success(f"Implied Vol: {sigma:.2%}")
        except Exception as e:
            st.sidebar.error("Calculation Error")
            sigma = 0.20
else:
    # Manual Input Mode
    vol_input = st.sidebar.slider("Volatility (σ %)", 10.0, 150.0, value = st.session_state.vol_val, step=0.5)
    sigma = vol_input / 100
st.sidebar.markdown("---")
st.sidebar.header("💼 Portfolio Position")
position_size = st.sidebar.number_input("Number of Contracts", value=10, step=1)
confidence_level = st.sidebar.number_input("VaR Confidence Level", min_value=0.01, max_value=0.999, value=0.95, step=0.01,format="%.3f")
K = st.number_input(
    "Strike Price ($)", 
    value=float(st.session_state.strike_k), 
    step=1.0, 
    key='input_strike_widget', 
    on_change=update_strike
)

# --- MAIN LOGIC ---
if current_price:
    # --- ENGINE: CALCULATE ALL MODELS ---
    with st.spinner('Engines running: Computing all models...'):
        # Convert option type to lowercase for function compatibility
        opt_type = option_type.lower()
        
        # Black-Scholes (European Baseline)
        t0 = time.time()
        bs_price = black_scholes(current_price, K, T, r, q, sigma, opt_type)
        time_bs = (time.time() - t0) * 1000

        # Monte Carlo European
        t0 = time.time()
        mc_euro_price, paths = monte_carlo_simulation(current_price, K, T, r, q, sigma, N=500, option_type=opt_type)
        time_mc_euro = (time.time() - t0) * 1000

        # Bjerksund-Stensland (American Analytic)
        t0 = time.time()
        am_bj_price = bjerksund_stensland_2002(current_price, K, T, r, q, sigma, opt_type)
        time_bj = (time.time() - t0) * 1000

        # Binomial & Trinomial Trees (Numerical Methods)
        t0 = time.time()
        am_bin_price = binomial_cached(current_price, K, T, r, q, sigma, N=n_steps, opt_type=opt_type)
        time_bin = (time.time() - t0) * 1000

        t0 = time.time()
        am_tri_price = trinomial_cached(current_price, K, T, r, q, sigma, N=n_steps, opt_type=opt_type)
        time_tri = (time.time() - t0) * 1000

        # FDM (PDE Solver - Finite Difference Method)
        t0 = time.time()
        am_pde_price = pde_cached(current_price, K, T, r, q, sigma, M=100, N=n_steps, opt_type=opt_type)
        time_pde = (time.time() - t0) * 1000

        # Monte Carlo American (Longstaff-Schwartz)
        t0 = time.time()
        mc_amer_price = mc_amer_cached(current_price, K, T, r, q, sigma, N_steps=n_steps, N_sims=500, opt_type=opt_type)
        time_mc_amer = (time.time() - t0) * 1000

    # --- HEADER : VALUATION  ---
    st.markdown("### Valuation & Market Data")

    st.metric("Spot Price (S)", f"${current_price:.2f}", help="Latest closing price retrieved from Yahoo Finance.")
    # On crée 5 colonnes pour tout voir d'un coup
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("European (BS)", f"${bs_price:.2f}", help="Theoretical European price (Black-Scholes)")

    with m2:
        # On affiche le prix américain le plus robuste (B&S ou PDE)
        premium = am_bj_price - bs_price
        st.metric("American (B&S)", f"${am_bj_price:.2f}", delta=f"{premium:.4f} Premium")

    with m3:
        st.metric("Implied Vol (σ)", f"{sigma:.2%}", delta=f"{sigma - hist_vol:.2%} vs Hist", delta_color="inverse")
    with m4:
        st.metric("Dividend Yield (q)", f"{q:.2%}")

    st.markdown("---")


    # --- TECHNICAL BENCHMARK TABLE ---
    with st.expander(" View Technical Model Comparison ", expanded=False):
        bench_data = {
            "Model": ["Black-Scholes", "Monte Carlo European", "Bjerksund-Stensland", "Binomial Tree", "Trinomial Tree", "Finite Difference (PDE)", "Monte Carlo American"],
            "Style": ["European", "European", "American", "American", "American", "American", "American"],
            "Price ($)": [bs_price, mc_euro_price, am_bj_price, am_bin_price, am_tri_price, am_pde_price, mc_amer_price],
            "Premium vs Euro": [0.0, mc_euro_price - bs_price, am_bj_price - bs_price, am_bin_price - bs_price, am_tri_price - bs_price, am_pde_price - bs_price, mc_amer_price - bs_price],
            "Execution Time": [f"{time_bs:.2f}ms", f"{time_mc_euro:.2f}ms", f"{time_bj:.2f}ms", f"{time_bin:.2f}ms", f"{time_tri:.2f}ms", f"{time_pde:.2f}ms", f"{time_mc_amer:.2f}ms"]
        }
        df_bench = pd.DataFrame(bench_data)
        st.table(df_bench.style.format({"Price ($)": "{:.4f}", "Premium vs Euro": "{:+.4f}"}))
        st.caption("Monte-Carlo: 500 simulations")
        st.caption("*Note: The first Bjerksund-Stensland calculation may include a slight 'cold start' overhead. Refresh the page to witness its true, real-time speed.*")

    st.markdown("---")

    delta, gamma, vega, theta, rho = calculate_greeks(current_price, K, T, r, q, sigma, option_type)

    # --- TAB DEFINITIONS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Strategy",
        " Greeks",
        " P&L Analysis", 
        " Monte Carlo", 
        " Risk Management"
    ])

    # --- TAB CONTENT ---

    # TAB 1: STRATEGY 
    with tab1:
        st.subheader(" Optimal Exercise Strategy")
        t_boundary, p_boundary = calculate_exercise_boundary(current_price, K, T, r, q, sigma, opt_type)
        is_nan = np.isnan(p_boundary[-1])
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown(f"""
            **How to read this chart:**
            - The **Red Dashed Line** is the decision boundary.
            - If the stock price enters the **shaded region**, you should **Exercise Immediately**.
            - In the white region, it is better to **Hold** the option.
            """)

            if is_nan:
                st.warning("⚠️ No exercise boundary detected.")
                st.write("For a Call without dividends, early exercise is **never** optimal. It is usually better to sell the option than to exercise it.")
            else:
                current_b = p_boundary[-1]
                dist = abs(current_price - current_b)
                st.metric("Boundary Level", f"${current_b:.2f}")
                st.write(f"Distance to current price: `${dist:.2f}`")
            
                if (opt_type == "call" and current_price >= current_b) or \
                (opt_type == "put" and current_price <= current_b):
                    st.error(" STATUS: IMMEDIATE EXERCISE RECOMMENDED")
                else:
                    st.success("✅ STATUS: HOLD")

        with col_b:
            if not is_nan:
                fig_bound, ax_bound = plt.subplots(figsize=(10, 5))
        
                # Reverse time array to show the approach towards maturity (T -> 0)
                time_to_expiry = t_boundary[::-1] 
        
                ax_bound.plot(time_to_expiry, p_boundary, 'r--', label="Exercise Boundary ($I_t$)", linewidth=2)
                ax_bound.axhline(y=current_price, color='blue', label="Current Spot Price", alpha=0.6)
        
                # Zone Shading
                if option_type == "Call":
                    ax_bound.fill_between(time_to_expiry, p_boundary, max(p_boundary)*1.2, color='red', alpha=0.1, label="Exercise Zone")
                else:
                    ax_bound.fill_between(time_to_expiry, 0, p_boundary, color='red', alpha=0.1, label="Exercise Zone")
            
                ax_bound.set_xlabel("Time to Expiry (Years)")
                ax_bound.set_ylabel("Stock Price ($)")
                ax_bound.set_title(f"Optimal Exercise Frontier ({option_type} American)")
                ax_bound.legend()
                ax_bound.invert_xaxis() # Invert axis to visualize time passing towards the left
        
                st.pyplot(fig_bound)
                plt.close(fig_bound)


    # TAB 2: GREEKS VISUALIZATION
    with tab2:
        st.subheader("Sensitivity (Greeks)")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{delta:.3f}", help="Change in option price for $1 change in stock.")
        g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Rate of change of Delta.")
        g3.metric("Vega (ν)", f"{vega:.3f}", help="Change in price for 1% change in volatility.")
        g4.metric("Theta (Θ)", f"{theta:.3f}", help="Time decay per day.")
        g5.metric("Rho (ρ)", f"{rho:.3f}", help="Sensitivity to interest rates.")
        st.subheader("Greeks Risk Profile")
        
        #  Visualization Row
        spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        

        deltas_plot, gammas_plot, vegas_plot, thetas_plot, rhos_plot = calculate_greeks(spot_range, K, T, r, q, sigma, option_type)

        fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Delta Plot
        axes[0, 0].plot(spot_range, deltas_plot, color='blue', linewidth=2)
        axes[0, 0].set_title("Delta (Directional Risk)")
        axes[0, 0].axvline(x=current_price, color='black', linestyle='--', alpha=0.5, label="Current Spot")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel("Delta Value")

        # Gamma Plot
        axes[0, 1].plot(spot_range, gammas_plot, color='orange', linewidth=2)
        axes[0, 1].set_title("Gamma (Convexity Risk)")
        axes[0, 1].axvline(x=current_price, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # Theta Plot
        axes[1, 0].plot(spot_range, thetas_plot, color='red', linewidth=2)
        axes[1, 0].set_title("Theta (Time Decay)")
        axes[1, 0].axvline(x=current_price, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylabel("Daily Value Loss ($)")

        # Vega Plot
        axes[1, 1].plot(spot_range, vegas_plot, color='green', linewidth=2)
        axes[1, 1].set_title("Vega (Volatility Risk)")
        axes[1, 1].axvline(x=current_price, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()
        st.subheader("Greeks 3D Surface (Heatmap)")
        st.markdown("Visualize how sensitivity evolves as a function of **Price** and **Time** simultaneously.")

        # UI parameters
        col_heat1, col_heat2 = st.columns(2)
        greek_choice = col_heat1.selectbox("Greek to visualize", ["Delta", "Gamma", "Theta", "Vega", "Rho"])
        
        v_spot, v_time, d_g, g_g, v_g, t_g, r_g = calculate_greeks_surface(current_price, K, T, r, q, sigma, option_type)

        # Map selection to the correct result matrix
        if greek_choice == "Delta": Z = d_g
        elif greek_choice == "Gamma": Z = g_g
        elif greek_choice == "Vega": Z = v_g
        elif greek_choice == "Theta": Z = t_g
        elif greek_choice == "Rho":  Z = r_g

        # Heatmap visualization
        fig_heat, ax_heat = plt.subplots(figsize=(12, 7))
        custom_cmap = "RdBu" if greek_choice in ["Delta", "Rho"] else "viridis"
        
        sns.heatmap(
            Z, 
            xticklabels=np.round(v_spot, 1), 
            yticklabels=np.round(v_time, 2), 
            annot=True, fmt=".2f", cmap=custom_cmap,
            cbar_kws={'label': f'{greek_choice} Value'},
            ax=ax_heat
        )
        
        ax_heat.invert_yaxis() # Invert to show longer time at the top
        ax_heat.set_title(f"Sensitivity Surface: {greek_choice}", fontsize=15)
        ax_heat.set_xlabel("Stock Price ($)")
        ax_heat.set_ylabel("Time to Expiration (Years)")
        
        st.pyplot(fig_heat)
        plt.close(fig_heat)
        st.info("💡 **How to read:** Higher color intensity indicates higher sensitivity to the parameter.")

    # TAB 3: P&L
    with tab3:
        st.subheader("Profit & Loss Analysis")
    
        # Calculate Break-even Point
        if option_type == "Call":
            breakeven = K + am_bj_price
        else:
            breakeven = K - am_bj_price
        
        c1, c2 = st.columns(2)
        c1.metric("Break-even Price", f"${breakeven:.2f}")
        c2.metric("Distance to Break-even", f"{((breakeven/current_price)-1):.2%}")

        # --- P&L Chart ---
        s_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
        # At expiration, the American price converges to the European payoff
        final_payoff = np.maximum(s_range - K, 0) if option_type == "Call" else np.maximum(K - s_range, 0)
    
        # Calculate P&L (Multiplier: 1 contract = 100 shares)
        pnl = (final_payoff - am_bj_price) * position_size * 100 
    
        fig_pnl, ax_pnl = plt.subplots(figsize=(10, 4))
        ax_pnl.plot(s_range, pnl, color='blue', linewidth=2)

        pnl_today = calculate_pnl_grid(s_range, K, T, r, q, sigma, opt_type, am_bj_price, position_size)

        ax_pnl.plot(s_range, pnl_today, color='cyan', linestyle='--', label="Current P&L (Today)")
    
        # Color shading for Profit (Green) and Loss (Red) zones
        ax_pnl.fill_between(s_range, pnl, 0, where=(pnl > 0), color='green', alpha=0.3)
        ax_pnl.fill_between(s_range, pnl, 0, where=(pnl < 0), color='red', alpha=0.3)
    
        ax_pnl.axhline(0, color='black', linewidth=1)
        ax_pnl.axvline(breakeven, color='orange', linestyle='--', label=f"Break-even: ${breakeven:.2f}")
    
        ax_pnl.set_title("P&L Projection at Expiration")
        ax_pnl.set_xlabel("Stock Price at Expiration ($)")
        ax_pnl.set_ylabel("Profit / Loss ($)")
        ax_pnl.legend()
    
        st.pyplot(fig_pnl)
        plt.close(fig_pnl)

    # TAB 4: MONTE CARLO
    with tab4:
        st.subheader(" Monte-Carlo Simulation")
        
        col_mc1, col_mc2 = st.columns([1, 3])
        
        with col_mc1:
            st.markdown("### Simulation Settings")
            n_simulation = st.number_input("Number of Simulations", value=1000, step=500, min_value=100)
            
            if st.button("Run Simulation"):
                with st.spinner("Simulating market scenarios..."):
                    mc_euro_price, paths = monte_carlo_simulation(current_price, K, T, r, q, sigma, N=n_simulation, option_type=opt_type)
                    mc_amer_price = mc_amer_cached(current_price, K, T, r, q, sigma, N_steps=n_steps, N_sims=n_simulation, opt_type=opt_type)

                    # Store in session state to persist data across re-runs
                    st.session_state['mc_paths'] = paths
                    st.session_state['mc_euro_price'] = mc_euro_price
                    st.session_state['mc_amer_price'] = mc_amer_price
                    st.success("Simulation Updated!")

        with col_mc2:
            if 'mc_paths' in st.session_state:
                paths = st.session_state['mc_paths']
                mc_euro_price = st.session_state['mc_euro_price']
                mc_amer_price = st.session_state['mc_amer_price']
                
                fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
                time_axis = np.linspace(0, T, paths.shape[0])
                
                # Probability Cone 
                upper_cone = np.percentile(paths, 97.5, axis=1)
                lower_cone = np.percentile(paths, 2.5, axis=1)
                ax_mc.fill_between(time_axis, lower_cone, upper_cone, color='blue', alpha=0.1, label="95% Confidence Interval")
                
                # Individual Trajectories (Sample of first 30 paths)
                ax_mc.plot(time_axis, paths[:, :30], alpha=0.3, linewidth=0.8)
                
                # Mean (Expected Price Path)
                ax_mc.plot(time_axis, np.mean(paths, axis=1), color='navy', linewidth=2, label="Mean Path")
                
                ax_mc.axhline(K, color='red', linestyle='--', label=f"Strike ${K}")
                ax_mc.set_title(f"Market Scenarios: {paths.shape[1]} Stochastic Paths")
                ax_mc.set_xlabel("Time (Years)")
                ax_mc.set_ylabel("Stock Price ($)")
                ax_mc.legend(loc='upper left')
                st.pyplot(fig_mc)
                plt.close(fig_mc)
            else:
                st.info("Adjust settings on the left and click 'Run Simulation' to visualize the price paths.")
                
        # Comparison of Monte Carlo Methods
        if 'mc_paths' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Model Comparison (Monte Carlo Results)")
        
            c1, c2, c3 = st.columns(3)
        
            mc_euro = st.session_state['mc_euro_price']
            mc_amer = st.session_state['mc_amer_price']
            premium_mc = mc_amer - mc_euro
        
            c1.metric("European MC Price", f"${mc_euro:.3f}")
            c2.metric("American MC Price (LSM)", f"${mc_amer:.3f}")
        
            pct_diff = (premium_mc / mc_euro * 100) if mc_euro > 0 else 0
            c3.metric("Early Exercise Premium", f"${premium_mc:.4f}", delta=f"{pct_diff:.2f}%")

            st.caption(f"Note: The early exercise premium of **${premium_mc:.4f}** is the additional value of the American option compared to the European one, derived from {n_simulation} paths.")
            
    # TAB 5: RISK
    with tab5:
        st.subheader(f" Value at Risk (VaR) - {confidence_level*100:.1f}% Confidence Level")

        if 'mc_paths' in st.session_state:
            paths = st.session_state['mc_paths']
            
            multiplier = 100
            # Datat preparation
            var_unit, pnl_dist_unit = calculate_var(paths, K, T, r, q, sigma, option_type, confidence_level)
            
            var_portfolio = var_unit * position_size * multiplier
            pnl_dist_portfolio = pnl_dist_unit * position_size * multiplier
            

            total_invested = am_bj_price * position_size * multiplier

            # --- 2. KPI DISPLAY (The "Bloomberg" View) ---
            st.markdown("####  Portfolio Risk Metrics")
            
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            
            # Total Exposure
            col_kpi1.metric(
                label="Total Invested Premium",
                value=f"${total_invested:,.2f}",
                help="Total amount currently at risk (Price * Position)."
            )
            
            # VaR
            col_kpi2.metric(
                label=f"VaR ({confidence_level*100:.1f}%)",
                value=f"${var_portfolio:,.2f}",
                delta=f"{(var_portfolio/total_invested)*100:.1f}% of capital",
                delta_color="inverse",
                help=f"Minimum loss expected in the worst {(1-confidence_level)*100:.1f}% of cases."
            )
            
            # Worst Case in Simulation
            worst_case = np.min(pnl_dist_portfolio)
            col_kpi3.metric(
                label="Worst Simulated Scenario",
                value=f"${worst_case:,.2f}",
                delta="Potential Total Loss",
                delta_color="inverse",
                help="The absolute worst outcome observed in the simulation."
            )

            st.divider()

            # ---  VISUALIZATION ---
            st.subheader("Distribution of Potential P&L at Maturity")

            fig_var, ax_var = plt.subplots(figsize=(12, 6))

            # Probability of Profit
            pop = np.mean(pnl_dist_portfolio > 0)
            col_kpi1.metric("Prob. of Profit (POP)", f"{pop:.1%}", help="Percentage of simulations that end in profit.")

            # Histogram
            n, bins, patches = ax_var.hist(
                pnl_dist_portfolio, 
                bins=70, 
                edgecolor='black', 
                linewidth=0.5, 
                alpha=0.7
            )

            for i in range(len(patches)):
                bin_center = (bins[i] + bins[i+1]) / 2
                
                if bin_center < var_portfolio:
                    patches[i].set_facecolor('#8B0000') # Dark Red (Extreme Risk)
                elif bin_center < 0:
                    patches[i].set_facecolor('#FF6F61') # Light Red (Loss)
                else:
                    patches[i].set_facecolor('#2E8B57') # Sea Green (Profit)

            # Add Vertical Lines for Context
            ax_var.axvline(x=0, color='black', linewidth=2, linestyle='-', label="Breakeven ($0)")
            ax_var.axvline(x=var_portfolio, color='red', linewidth=2, linestyle='--', label=f"VaR Threshold")
            
            # Formatting the Chart
            ax_var.set_title(f"Projected Profit & Loss (P&L) for {position_size} Contracts", fontsize=14)
            ax_var.set_xlabel("Profit / Loss ($)", fontsize=12)
            ax_var.set_ylabel("Frequency (Number of Scenarios)", fontsize=12)
            ax_var.legend(loc='upper right')
            ax_var.grid(True, alpha=0.2, linestyle='--')

            st.pyplot(fig_var)
            plt.close(fig_var)
            
            # Contextual Text
            st.info(
                f"Interpretation:** The red zone represents the scenarios where you lose money. "
                f"The dark red tail on the left shows the extreme risk (VaR). "
                f"The green zone represents profitable scenarios."
            )

        else:
            st.warning("⚠️ No simulation data found. Please go to the 'Monte Carlo' tab and run the simulation first.")
        st.divider()
        
        # --- HEDGING STRATEGY SECTION ---
        st.subheader(" Delta Hedging Strategy")
        st.markdown("How to immunize your portfolio against small variations in the underlying stock price?")

        # Calculate shares needed to reach Delta-Neutrality (Delta * Contracts * 100)
        shares_to_hedge = delta * position_size * 100
        qty = abs(shares_to_hedge)
        
        col_hedge1, col_hedge2 = st.columns([1.5, 1])
        
        with col_hedge1:
            if option_type == "Call":
                side = "SELL (Short)"
                action_color = "red"
                explanation = "Your Call gains value when the stock rises. To offset this gain (and the associated risk), you must sell shares."
            else:
                side = "BUY (Long)"
                action_color = "green"
                explanation = "Your Put gains value when the stock falls. To compensate, you must hold shares that rise in value."

            # Trading Floor style recommendation box
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa; 
                border-left: 5px solid {action_color}; 
                padding: 15px; 
                border-radius: 5px;">
                <h4 style="margin:0; color: #333;">Delta-Neutral Recommendation:</h4>
                <p style="font-size: 18px; margin: 10px 0;">
                    You must <b>{side}</b> exactly <b style="color:{action_color}; font-size: 24px;">{qty:,.0f}</b> shares of {ticker}.
                </p>
                <p style="font-size: 12px; color: #666; margin:0;">
                    <i>{explanation}</i>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_hedge2:
            # Summary table for the calculation
            st.markdown("#### Calculation Details")
            st.markdown(f"""
            | Parameter | Value |
            | :--- | :--- |
            | Unit Delta | `{delta:.4f}` |
            | Position (Contracts) | `{position_size}` |
            | Multiplier | `x 100` |
            | **Total Shares** | **`{qty:,.1f}`** |
            """)

else:
    st.warning("Please enter a valid Ticker.")
