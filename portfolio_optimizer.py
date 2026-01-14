import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Quantitative Portfolio Optimizer", layout="wide", page_icon="📈")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Parameters")

# Ticker Input
default_tickers = "AAPL, MSFT, GOOG, AMZN, TSLA, TLT"
tickers_input = st.sidebar.text_input("Assets (Comma Separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

# The Critical Variables
st.sidebar.markdown("---")
st.sidebar.subheader("Market Assumptions")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Annual %)", 0.0, 15.0, 4.0, 0.5) / 100
include_cash = st.sidebar.checkbox("Include 'Risk-Free Cash' as an Investable Asset?", value=True)

st.sidebar.info(
    """
    **Tip:** If you check the box above, the optimizer can choose to hold Cash (yielding 4%) 
    instead of risky stocks if the market volatility is too high.
    """
)

# --- 3. CORE FUNCTIONS (The Engine) ---
@st.cache_data
def get_market_data(ticker_list, start):
    """
    Fetches historical data and handles the 'Adj Close' vs 'Close' reliability issue.
    """
    try:
        data = yf.download(ticker_list, start=start, progress=False)
        
        # Prefer Adjusted Close (accounts for dividends/splits)
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            return pd.DataFrame() # Return empty if failed

        # Handle MultiIndex columns if necessary
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
            
        return prices.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def run_monte_carlo(mean_rets, cov_mat, rf_rate, n_sims=5000):
    """
    Runs N random simulations to map the Efficient Frontier.
    """
    num_assets = len(mean_rets)
    results = np.zeros((3, n_sims)) # Rows: [Return, Volatility, Sharpe]
    weights_record = []

    for i in range(n_sims):
        # Generate Random Weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Portfolio Maths
        p_ret = np.sum(mean_rets * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
        
        # Store Results
        results[0,i] = p_ret
        results[1,i] = p_vol
        results[2,i] = (p_ret - rf_rate) / p_vol # Sharpe Ratio

    return results, weights_record

# --- 4. MAIN APP LOGIC ---
st.title("Modern Portfolio Theory (MPT) Visualizer")
st.markdown("Optimize asset allocation using Monte Carlo simulations to maximize the **Sharpe Ratio**.")

if st.sidebar.button("Run Optimization", type="primary"):
    
    with st.spinner("Crunching numbers, simulating markets, and optimizing allocations..."):
        
        # A. Fetch Data
        raw_prices = get_market_data(tickers, start_date)
        
        if raw_prices.empty:
            st.error("No data found. Please check your ticker symbols.")
            st.stop()

        # B. Prepare Returns
        daily_returns = raw_prices.pct_change().dropna()

        # C. (Optional) Inject Synthetic Cash
        if include_cash:
            # Daily return equivalent of the annual risk-free rate
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            daily_returns['CASH'] = daily_rf
        
        # D. Calculate Statistics
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        sim_tickers = daily_returns.columns.tolist()

        # E. Run Simulation
        sim_results, sim_weights = run_monte_carlo(mean_returns, cov_matrix, risk_free_rate)
        
        # Convert to DataFrame for Plotting
        sim_df = pd.DataFrame(sim_results.T, columns=['Return', 'Volatility', 'Sharpe'])
        
        # F. Find the Winner
        max_sharpe_idx = sim_df['Sharpe'].idxmax()
        best_port = sim_df.iloc[max_sharpe_idx]
        best_weights = sim_weights[max_sharpe_idx]

        # --- 5. VISUALIZATION DASHBOARD ---
        
        # Create Tabs for Organization
        tab1, tab2, tab3 = st.tabs(["📊 Optimization Results", "🔎 Asset Analysis", "🎓 Methodology"])

        # TAB 1: THE RESULTS
        with tab1:
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.subheader("The Efficient Frontier")
                fig_ef = px.scatter(sim_df, x='Volatility', y='Return', color='Sharpe',
                                    title="Risk vs. Return (5,000 Scenarios)",
                                    labels={'Volatility': 'Annual Risk (Std Dev)', 'Return': 'Annual Expected Return'},
                                    color_continuous_scale='Bluered_r')
                
                # Highlight the Optimal Portfolio
                fig_ef.add_scatter(x=[best_port['Volatility']], y=[best_port['Return']], 
                                   mode='markers', marker=dict(color='gold', size=20, symbol='star', line=dict(width=2, color='black')),
                                   name=f"Optimal Portfolio (Sharpe: {best_port['Sharpe']:.2f})")
                
                st.plotly_chart(fig_ef, use_container_width=True, key="ef_chart")

            with col_res2:
                st.subheader("Optimal Allocation")
                
                # Metric Cards
                c1, c2 = st.columns(2)
                c1.metric("Expected Return", f"{best_port['Return']:.1%}")
                c2.metric("Annual Risk", f"{best_port['Volatility']:.1%}")
                st.metric("Sharpe Ratio", f"{best_port['Sharpe']:.2f}")
                
                # Pie Chart
                alloc_df = pd.DataFrame({'Asset': sim_tickers, 'Weight': best_weights})
                # Filter small weights for cleanliness
                alloc_df = alloc_df[alloc_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
                
                fig_pie = px.pie(alloc_df, values='Weight', names='Asset', 
                                 title="Suggested Portfolio Weights", hole=0.4)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

        # TAB 2: DEEP DIVE
        with tab2:
            st.subheader("Correlation Matrix")
            st.markdown("This heatmap shows how assets move in relation to each other. **Light colors (near 0)** or **Blue (negative)** indicate good diversification partners.")
            
            corr_matrix = daily_returns.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_chart")
            
            st.subheader("Individual Asset Risk/Return Profile")
            
            # --- FIX: ROBUST COLUMN NAMING ---
            asset_stats = pd.DataFrame({
                "Annual Return": mean_returns * 252,
                "Annual Volatility": daily_returns.std() * np.sqrt(252)
            })
            # Force reset index and explicitly name columns so 'text='Asset' always works
            asset_stats = asset_stats.reset_index()
            asset_stats.columns = ['Asset', 'Annual Return', 'Annual Volatility']
            
            fig_bar = px.scatter(asset_stats, x='Annual Volatility', y='Annual Return', text='Asset', size_max=60,
                                 title="Stock vs. Stock Comparison")
            fig_bar.update_traces(textposition='top center')
            st.plotly_chart(fig_bar, use_container_width=True, key="scatter_assets")

        # TAB 3: METHODOLOGY
        with tab3:
            st.subheader("How this works")
            st.markdown("""
            This application uses **Modern Portfolio Theory (MPT)** to find the mathematically optimal mix of assets.
            
            **1. The Sharpe Ratio**
            We optimize for the risk-adjusted return using the Sharpe Ratio formula:
            """)
            st.latex(r'''
            Sharpe = \frac{R_p - R_f}{\sigma_p}
            ''')
            st.markdown(f"""
            Where:
            * $R_p$ is the Portfolio Return
            * $R_f$ is the Risk-Free Rate (currently set to **{risk_free_rate*100}%**)
            * $\sigma_p$ is the Portfolio Standard Deviation (Risk)
            
            **2. Monte Carlo Simulation**
            Instead of solving complex quadratic equations, we simulate **5,000 random portfolios**. 
            By plotting every possible combination of weights, we reveal the "Efficient Frontier"—the boundary where you cannot get more return without taking more risk.
            """)

            st.markdown("---")
            st.subheader("⚠️ Limitations of MPT")
            st.markdown("""
            While MPT is a Nobel Prize-winning theory, it is not without flaws. Acknowledging these is crucial for real-world application:
            
            1.  **Historical Bias:** The model assumes that *future* returns and correlations will look exactly like the *past*. In reality, market regimes change (e.g., inflation spikes, tech bubbles burst).
            2.  **Normal Distribution Assumption:** MPT assumes returns follow a "Bell Curve." It often underestimates the probability of "Black Swan" events (extreme market crashes) where correlations converge to 1.
            3.  **Static Correlations:** The model assumes the relationship between assets (e.g., Stocks vs. Bonds) is constant. In times of panic, assets that usually move oppositely may crash together.
            """)

else:
    st.info("Use the sidebar to select stocks and click **Run Optimization** to start.")