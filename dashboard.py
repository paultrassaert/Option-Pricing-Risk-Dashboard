import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import get_data, black_scholes, calculate_greeks, monte_carlo_simulation, calculate_var, calculate_implied_volatility

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

# --- CONFIGURATION ---
st.set_page_config(page_title="Option Pricer and Risk", layout="wide", page_icon="⚡")
st.title("⚡ Option Pricing and Risk Dashboard")
st.markdown("Pricing • Simulation Monte Carlo • Value at Risk (VaR)")

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Market Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="TSLA")
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
T = st.sidebar.number_input("Time to Expiration (Years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.045, format="%.3f")

# --- DATA LOADING ---
try:
    current_price, hist_vol = get_data(ticker)
except Exception as e:
    st.error(f"Could not fetch data for {ticker}. Please check the symbol.")
    st.stop()

# SESSION STATE MANAGEMENT (STRIKE PRICE)
    
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ticker
    st.session_state.strike_k = float(current_price)

# Detect if Ticker changed since last run
if st.session_state.last_ticker != ticker:
    st.session_state.strike_k = float(current_price) 
    st.session_state.last_ticker = ticker

def update_strike():
    st.session_state.strike_k = st.session_state.input_strike_widget

# --- SIDEBAR INPUTS ---
use_implied_vol = st.sidebar.checkbox("Calibrate with Market Price (IV)")
if use_implied_vol:
    default_price = black_scholes(current_price, float(st.session_state.strike_k), T, r, 0.4, option_type)

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
            implied_vol = calculate_implied_volatility(market_price_input, current_price, K_val, T, r, option_type)
            sigma = implied_vol
            st.sidebar.success(f"Implied Vol: {sigma:.2%}")
        except Exception as e:
            st.sidebar.error("Calculation Error")
            sigma = 0.20
else:
    # Manual Input Mode
    vol_input = st.sidebar.slider("Volatility (σ %)", 10, 150, 40)
    sigma = vol_input / 100
st.sidebar.markdown("---")
st.sidebar.header("💼 Portfolio Position")
position_size = st.sidebar.number_input("Number of Contracts", value=10, step=1)
confidence_level = st.sidebar.number_input("VaR Confidence Level", min_value=0.01, max_value=0.999, value=0.95, step=0.01,format="%.3f")

# --- MAIN LOGIC ---
if current_price:
    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Historical Vol (1Y)", f"{hist_vol:.2%}")
    col3.metric("Vol Used for Pricing", f"{sigma:.2%}")
    
    st.markdown("---")

    # Inputs for Pricing
    c1, c2 = st.columns([1, 1])
    with c1:
        K = st.number_input("Strike Price ($)", 
                            value=float(st.session_state.strike_k), 
                            step=1.0, 
                            key='input_strike_widget', 
                            on_change=update_strike)
    with c2:
        st.info(f"Volatility used for pricing: **{sigma:.2%}**")

    # --- CALCULATIONS ---
    # Black-Scholes model
    bs_price = black_scholes(current_price, K, T, r, sigma, option_type)
    
    # Calculate Greeks
    delta, gamma, vega, theta, rho = calculate_greeks(current_price, K, T, r, sigma, option_type)

    st.markdown("---")

    if option_type == "Call":
        card_color = "#dbfdde"
        text_color = "#006400" 
    else:
        card_color = "#ffcccc" 
        text_color = "#8b0000" 
    
    col_price, col_pnl = st.columns([1, 2])
    
    with col_price:
     
        st.markdown(f"""
        <div style="
            background-color: {card_color};
            padding: 20px;
            border-radius: 10px;
            border: 1px solid {text_color};
            text-align: center;
            box-shadow: 3px 3px 6px rgba(0,0,0,0.1);
            pointer-events: none; 
            user-select: none;"> 
            <h4 style="margin:0; color: {text_color}; opacity: 0.8;">{option_type} Black-Scholes Price</h4>
            <h1 style="margin:0; color: {text_color}; font-size: 34px;">${bs_price:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pnl:
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #cccccc;
            text-align: left;
            box-shadow: 3px 3px 6px rgba(0,0,0,0.05);
            pointer-events: none;
            user-select: none;">
            <h4 style="margin:0; color: #555;">Position Value ({position_size} Contracts)</h4>
            <h1 style="margin:0; color: #333; font-size: 34px;">${bs_price * position_size:,.2f}</h1>
            <p style="margin:0; color: #666; font-size: 14px;">Strike: <b>${K}</b> | Exp: <b>{T} Yr</b> | Vol: <b>{sigma*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")

    # --- TABS FOR DETAILED ANALYSIS ---
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Greeks", "📈 P&L Analysis", "🎲 Monte Carlo", "🛡️ Risk Management (VaR)"])

    # TAB 1: GREEKS
    with tab1:
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{delta:.3f}", help="Change in option price for $1 change in stock.")
        g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Rate of change of Delta.")
        g3.metric("Vega (ν)", f"{vega:.3f}", help="Change in price for 1% change in volatility.")
        g4.metric("Theta (Θ)", f"{theta:.3f}", help="Time decay per day.")
        g5.metric("Rho (ρ)", f"{rho:.3f}", help="Sensitivity to interest rates.")

        st.divider()
        st.subheader("Greeks Risk Profile")
        
        #  Visualization Row
        spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        

        deltas_plot, gammas_plot, vegas_plot, thetas_plot, rhos_plot = calculate_greeks(spot_range, K, T, r, sigma, option_type)

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

        st.divider()
        st.subheader("Greeks 3D Surface (Heatmap)")
        st.markdown("Visualize how sensitivity evolves as a function of **Price** and **Time** simultaneously.")

        # UI parameters
        col_heat1, col_heat2 = st.columns(2)
        greek_choice = col_heat1.selectbox("Greek to visualize", ["Delta", "Gamma", "Theta", "Vega", "Rho"])
        
        # Create grid: Spot Price (+/- 20%) vs Time to Expiration
        vol_spot = np.linspace(current_price * 0.8, current_price * 1.2, 15)
        vol_time = np.linspace(0.02, T, 15) 
        X, Y = np.meshgrid(vol_spot, vol_time)

        # Vectorized Greek calculation across the entire grid
        d_grid, g_grid, v_grid, t_grid, r_grid = calculate_greeks(X, K, Y, r, sigma, option_type)

        # Map selection to the correct result matrix
        if greek_choice == "Delta": Z = d_grid
        elif greek_choice == "Gamma": Z = g_grid
        elif greek_choice == "Vega": Z = v_grid
        elif greek_choice == "Theta": Z = t_grid
        elif greek_choice == "Rho":  Z = r_grid

        # Heatmap visualization
        fig_heat, ax_heat = plt.subplots(figsize=(12, 7))
        custom_cmap = "RdBu" if greek_choice in ["Delta", "Rho"] else "viridis"
        
        sns.heatmap(
            Z, 
            xticklabels=np.round(vol_spot, 1), 
            yticklabels=np.round(vol_time, 2), 
            annot=True, fmt=".2f", cmap=custom_cmap,
            cbar_kws={'label': f'{greek_choice} Value'},
            ax=ax_heat
        )
        
        ax_heat.invert_yaxis() # Invert to show longer time at the top
        ax_heat.set_title(f"Sensitivity Surface: {greek_choice}", fontsize=15)
        ax_heat.set_xlabel("Stock Price ($)")
        ax_heat.set_ylabel("Time to Expiration (Years)")
        
        st.pyplot(fig_heat)
        st.info("💡 **How to read:** Higher color intensity indicates higher sensitivity to the parameter.")

    # TAB 2: P&L CHART
    with tab2:
        st.subheader("Theoretical Price vs. Spot Price")
        
        # Create a range of spot prices +/- 50%
        spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        
        # Calculate option prices for this whole range 
        prices = black_scholes(spot_range, K, T, r, sigma, option_type)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(spot_range, prices, color="blue", linewidth=2, label="Value")
        
        # Add visual markers
        ax.axvline(x=current_price, color='black', linestyle='--', alpha=0.5, label="Current Price")
        ax.axvline(x=K, color='red', linestyle='--', alpha=0.5, label="Strike Price")
        
        ax.set_title(f"Option Price Sensitivity to Spot Price")
        ax.set_xlabel("Stock Price ($)")
        ax.set_ylabel("Option Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

    # TAB 3: MONTE CARLO
    with tab3:
        st.subheader("Monte Carlo Simulation")
        st.markdown("Simulating thousands of possible future price paths...")
        
        n_sims = st.number_input("Number of Simulations", value=1000, step=500, min_value=100)
        
        if st.button("🚀 Run Simulation"):
            with st.spinner("Simulating market scenarios..."):
                # Run the engine
                mc_price, paths = monte_carlo_simulation(current_price, K, T, r, sigma, N=n_sims, option_type=option_type)
                st.session_state['mc_paths'] = paths
                st.session_state['mc_price'] = mc_price
            
            # Display Results
            c1, c2 = st.columns(2)
            c1.metric("Monte Carlo Estimated Price", f"${mc_price:.2f}")
            c2.metric("Difference vs Black-Scholes", f"${(mc_price - bs_price):.2f}", 
                      delta_color="off" if abs(mc_price - bs_price) < 0.1 else "normal")
            
            # Plot 1: The Paths (First 50 only to avoid clutter)
            fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
            ax_mc.plot(paths[:, :50], alpha=0.3, linewidth=1, color='tab:blue')
            ax_mc.axhline(K, color='red', linestyle='--', label="Strike Price")
            ax_mc.set_title(f"Simulation of first 50 Paths")
            ax_mc.set_ylabel("Price ($)")
            ax_mc.set_xlabel("Time Steps (Days)")
            st.pyplot(fig_mc)
            
            # Plot 2: Distribution of Final Prices
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            final_prices = paths[-1]
            ax_hist.hist(final_prices, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax_hist.axvline(K, color='red', linestyle='--', label="Strike Price")
            ax_hist.set_title("Distribution of Final Prices at Maturity")
            ax_hist.set_xlabel("Final Price ($)")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)

            st.success("Simulation complete! Head to the 'Risk Management' tab to view the VaR.")
    
    # TAB 4 RISK MANAGEMENT
    with tab4:
        st.subheader(f"🛡️ Value at Risk (VaR) - {confidence_level*100:.1f}% Confidence Level")

        if 'mc_paths' in st.session_state:
            paths = st.session_state['mc_paths']
            
            # Datat preparation
            var_unit, pnl_dist_unit = calculate_var(paths, K, r, T, option_type, confidence_level)
            
            var_portfolio = var_unit * position_size
            pnl_dist_portfolio = pnl_dist_unit * position_size
            
            total_invested = bs_price * position_size

            # --- 2. KPI DISPLAY (The "Bloomberg" View) ---
            st.markdown("#### 📊 Portfolio Risk Metrics")
            
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
        st.subheader("⚖️ Delta Hedging Strategy")
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
            st.markdown("#### 🧮 Calculation Details")
            st.markdown(f"""
            | Parameter | Value |
            | :--- | :--- |
            | Unit Delta | `{delta:.4f}` |
            | Position (Contracts) | `{position_size}` |
            | Multiplier | `x 100` |
            | **Total Shares** | **`{qty:,.1f}`** |
            """)

else:
    st.warning("Please enter a valid ticker symbol.")