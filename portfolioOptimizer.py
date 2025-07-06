import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
import warnings
warnings.filterwarnings('ignore')

# Streamlit app configuration
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Modern Portfolio Theory - Portfolio Optimizer")
# Step 1: User Inputs
st.sidebar.header("Portfolio Parameters")
tickers_input = st.sidebar.text_input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL)", 
                                     value="AAPL,MSFT,GOOGL,AMZN,TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",")]
investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=1000, value=10000, step=1000)
start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
max_weight = st.sidebar.slider("Max Weight per Asset (%)", min_value=10.0, max_value=100.0, value=50.0, step=5.0) / 100

# Validate inputs
if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()
# Step 2: Data Collection
with st.spinner("Fetching data..."):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # Check for data issues
    if data.empty or data.isna().all().all():
        st.error("No data retrieved. Check tickers or internet connection.")
        st.stop()
    if data.isna().any().any():
        st.warning("Missing data detected. Filling with forward fill.")
        data = data.fillna(method='ffill').fillna(method='bfill')
    if len(data) < 100:
        st.error("Insufficient data points. Please select a longer date range.")
        st.stop()
# Step 3: Calculate Returns and Covariance
returns = data.pct_change().dropna()
mean_returns = expected_returns.mean_historical_return(data, frequency=252)
cov_matrix = risk_models.sample_cov(data, frequency=252)
# Step 4: Portfolio Optimization
try:
    ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, max_weight))
    
    # Max Sharpe Ratio Portfolio
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_weights = ef.clean_weights()
    max_sharpe_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    
    # Minimum Volatility Portfolio
    ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, max_weight))
    ef.min_volatility()
    min_vol_weights = ef.clean_weights()
    min_vol_perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
except Exception as e:
    st.error(f"Optimization failed: {str(e)}. Try different tickers or date range.")
    st.stop()
# Step 5: Discrete Allocation
latest_prices = data.iloc[-1]
da = DiscreteAllocation(max_sharpe_weights, latest_prices, total_portfolio_value=investment_amount)
allocation, leftover = da.greedy_portfolio()

# Step 6: Display Results
st.header("Optimal Portfolios")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Max Sharpe Ratio Portfolio")
    st.write(f"**Expected Return**: {max_sharpe_perf[0]:.2%}")
    st.write(f"**Volatility**: {max_sharpe_perf[1]:.2%}")
    st.write(f"**Sharpe Ratio**: {max_sharpe_perf[2]:.2f}")
    st.write("**Weights**:")
    for ticker, weight in max_sharpe_weights.items():
        st.write(f"{ticker}: {weight:.2%}")
    st.write("**Allocation**:")
    for ticker, shares in allocation.items():
        st.write(f"{ticker}: {shares} shares")
    st.write(f"**Funds Remaining**: ${leftover:.2f}")

with col2:
    st.subheader("Minimum Volatility Portfolio")
    st.write(f"**Expected Return**: {min_vol_perf[0]:.2%}")
    st.write(f"**Volatility**: {min_vol_perf[1]:.2%}")
    st.write(f"**Sharpe Ratio**: {min_vol_perf[2]:.2f}")
    st.write("**Weights**:")
    for ticker, weight in min_vol_weights.items():
        st.write(f"{ticker}: {weight:.2%}")
# Step 7: Visualization - Efficient Frontier
st.header("Efficient Frontier")
with st.spinner("Generating efficient frontier..."):
    try:
        ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, max_weight))
        n_points = 50  # Reduced for performance
        max_return = ef._max_return()
        frontier_returns = np.linspace(min_vol_perf[0], min(max_sharpe_perf[0], max_return * 0.99), n_points)
        frontier_vols = []
        for ret in frontier_returns:
            try:
                ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, max_weight))
                ef.efficient_return(ret)
                frontier_vols.append(ef.portfolio_performance(verbose=False)[1])
            except Exception:
                frontier_vols.append(np.nan)
        
        # Filter out NaN values
        valid_indices = ~np.isnan(frontier_vols)
        if not np.any(valid_indices):
            st.warning("No valid portfolio solutions for efficient frontier. Displaying max Sharpe and min volatility points.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[max_sharpe_perf[1], min_vol_perf[1]],
                y=[max_sharpe_perf[0], min_vol_perf[0]],
                mode='markers',
                name='Portfolios',
                marker=dict(size=15, symbol='star', color=['red', 'green']),
                text=['Max Sharpe', 'Min Volatility']
            ))
        else:
            frontier_returns = frontier_returns[valid_indices]
            frontier_vols = np.array(frontier_vols)[valid_indices]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frontier_vols, y=frontier_returns,
                mode='lines', name='Efficient Frontier',
                line=dict(color='blue'),
                hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[max_sharpe_perf[1]], y=[max_sharpe_perf[0]],
                mode='markers', name='Max Sharpe',
                marker=dict(symbol='star', size=15, color='red')
            ))
            fig.add_trace(go.Scatter(
                x=[min_vol_perf[1]], y=[min_vol_perf[0]],
                mode='markers', name='Min Volatility',
                marker=dict(symbol='star', size=15, color='green')
            ))
        fig.update_layout(
            title="Portfolio Optimization - Efficient Frontier",
            xaxis_title="Volatility (Standard Deviation)",
            yaxis_title="Expected Return",
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to generate efficient frontier: {str(e)}. Try adjusting parameters or tickers.")
        st.stop()
        
# Step 8: Download Report
st.header("Download Report")
report = pd.DataFrame({
    "Ticker": list(max_sharpe_weights.keys()),
    "Weight (%)": [w * 100 for w in max_sharpe_weights.values()],
    "Shares": [allocation.get(t, 0) for t in max_sharpe_weights.keys()]
})
report["Return (%)"] = mean_returns * 100
report["Volatility (%)"] = np.sqrt(np.diag(cov_matrix)) * 100
csv = report.to_csv(index=False)
st.download_button("Download Portfolio Report (CSV)", csv, "portfolio_report.csv", "text/csv")
