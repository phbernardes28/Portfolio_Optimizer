import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Quantitative Portfolio Optimizer", layout="wide", page_icon="📈")

# Custom CSS
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

default_tickers = "AAPL, MSFT, GOOG, AMZN, TSLA, TLT"
tickers_input = st.sidebar.text_input("Assets (Comma Separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

st.sidebar.markdown("---")
st.sidebar.subheader("Market Assumptions")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Annual %)", 0.0, 15.0, 4.0, 0.5) / 100
include_cash = st.sidebar.checkbox("Include 'Risk-Free Cash'?", value=True)

# --- 3. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_list, start):
    try:
        # Download data
        data = yf.download(ticker_list, start=start, progress=False, group_by='column')
        
        # Robust Logic for "Adj Close" vs "Close"
        if isinstance(data.columns, pd.MultiIndex):
            # If MultiIndex, check if 'Adj Close' exists in the top level
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        else:
            # If single index (rare with list, but possible with 1 ticker)
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

        # Force DataFrame format even if single ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        return prices.dropna()
    except Exception as e:
        return pd.DataFrame()

def run_monte_carlo(mean_rets, cov_mat, rf_rate, n_sims=5000):
    num_assets = len(mean_rets)
    results = np.zeros((3, n_sims)) 
    weights_record = []

    for i in range(n_sims):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Portfolio Maths
        p_ret = np.sum(mean_rets * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
        
        results[0,i] = p_ret
        results[1,i] = p_vol
        results[2,i] = (p_ret - rf_rate) / p_vol 

    return results, weights_record

# --- 4. MAIN APP LOGIC ---
st.title("Modern Portfolio Theory (MPT) Visualizer")

# Initialize Session State variables if they don't exist
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None

# BUTTON: Only triggers the calculation
if st.sidebar.button("Run Optimization", type="primary"):
    
    if len(tickers) < 2 and not include_cash:
        st.error("Please enter at least 2 assets to optimize a portfolio.")
        st.stop()
        
    with st.spinner("Crunching numbers..."):
        # A. Fetch Data
        raw_prices = get_market_data(tickers, start_date)
        
        if raw_prices.empty or raw_prices.shape[1] == 0:
            st.error("No data found. Check ticker symbols.")
        else:
            # B. Prepare Returns
            daily_returns = raw_prices.pct_change().dropna()

            # C. Inject Cash (The safe asset)
            if include_cash:
                daily_rf = (1 + risk_free_rate) ** (1/252) - 1
                daily_returns['CASH'] = daily_rf

            # Check if we have enough columns for covariance
            if daily_returns.shape[1] < 2:
                 st.error("Need at least 2 valid assets (including Cash) to calculate covariance.")
            else:
                # D. Statistics
                mean_returns = daily_returns.mean()
                cov_matrix = daily_returns.cov()
                
                # E. Run Simulation
                results, weights = run_monte_carlo(mean_returns, cov_matrix, risk_free_rate)
                
                # F. Store in Session State
                sim_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])
                
                # Find Optimal
                max_idx = sim_df['Sharpe'].idxmax()
                best_port = sim_df.iloc[max_idx]
                best_weights = weights[max_idx]
                
                # Save everything to state
                st.session_state.sim_data = {
                    'sim_df': sim_df,
                    'best_port': best_port,
                    'best_weights': best_weights,
                    'tickers': daily_returns.columns.tolist(),
                    'cov_matrix': cov_matrix,
                    'mean_returns': mean_returns
                }
                st.session_state.sim_results = True

# --- 5. VISUALIZATION (Renders from Session State) ---
if st.session_state.sim_results:
    
    data = st.session_state.sim_data
    sim_df = data['sim_df']
    best_port = data['best_port']
    best_weights = data['best_weights']
    sim_tickers = data['tickers']

    tab1, tab2, tab3 = st.tabs(["📊 Optimization Results", "🔎 Asset Analysis", "🎓 Methodology"])

    with tab1:
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.subheader("The Efficient Frontier")
            fig_ef = px.scatter(sim_df, x='Volatility', y='Return', color='Sharpe',
                                title="Risk vs. Return (5,000 Scenarios)",
                                color_continuous_scale='Bluered_r')
            
            fig_ef.add_scatter(x=[best_port['Volatility']], y=[best_port['Return']], 
                               mode='markers', marker=dict(color='gold', size=20, symbol='star', line=dict(width=2, color='black')),
                               name=f"Optimal Portfolio (Sharpe: {best_port['Sharpe']:.2f})")
            
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_res2:
            st.subheader("Optimal Allocation")
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{best_port['Return']:.1%}")
            c2.metric("Risk", f"{best_port['Volatility']:.1%}")
            st.metric("Sharpe Ratio", f"{best_port['Sharpe']:.2f}")
            
            alloc_df = pd.DataFrame({'Asset': sim_tickers, 'Weight': best_weights})
            alloc_df = alloc_df[alloc_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("Correlation Matrix")
        fig_corr = px.imshow(data['cov_matrix'].corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Asset Risk/Return Profile")
        
        # --- FIXED COLUMN NAMING ---
        asset_stats = pd.DataFrame({
            "Annual Return": data['mean_returns'] * 252,
            "Annual Volatility": np.sqrt(np.diag(data['cov_matrix']) * 252)
        })
        asset_stats = asset_stats.reset_index()
        asset_stats.columns = ['Asset', 'Annual Return', 'Annual Volatility']
        
        fig_bar = px.scatter(asset_stats, x='Annual Volatility', y='Annual Return', text='Asset', size_max=60)
        fig_bar.update_traces(textposition='top center')
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("How this works")
        st.markdown("""
        This application uses **Modern Portfolio Theory (MPT)** to find the mathematically optimal mix of assets.
        
        ### 1. The Sharpe Ratio
        We optimize for the risk-adjusted return using the Sharpe Ratio formula:
        """)
        
        st.latex(r''' Sharpe = \frac{R_p - R_f}{\sigma_p} ''')
        
        st.markdown(f"""
        Where:
        * $R_p$ is the Portfolio Return
        * $R_f$ is the Risk-Free Rate (currently set to **{risk_free_rate*100}%**)
        * $\sigma_p$ is the Portfolio Standard Deviation (Risk)
        
        ### 2. Monte Carlo Simulation
        Instead of solving complex quadratic equations, we simulate **5,000 random portfolios**. 
        By plotting every possible combination of weights, we reveal the **"Efficient Frontier"**—the boundary where you cannot get more return without taking more risk.
        """)

        st.warning("""
        ### ⚠️ Limitations of MPT
        While MPT is a Nobel Prize-winning theory, it is not without flaws. Acknowledging these is crucial for real-world application:
        
        * **Historical Bias:** The model assumes that *future* returns and correlations will look exactly like the *past*. In reality, market regimes change (e.g., inflation spikes, tech bubbles burst).
        * **Normal Distribution Assumption:** MPT assumes returns follow a "Bell Curve." It often underestimates the probability of "Black Swan" events (extreme market crashes) where correlations converge to 1.
        * **Static Correlations:** The model assumes the relationship between assets (e.g., Stocks vs. Bonds) is constant. In times of panic, assets that usually move oppositely may crash together.
        """)
else:
    st.info("👈 Enter tickers in the sidebar and click **Run Optimization**.")
