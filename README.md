# Portfolio Optimizer using Modern Portfolio Theory (MPT)

This project is a **Streamlit web application** that helps investors optimize their portfolio using **Modern Portfolio Theory (MPT)**.  
It leverages **historical stock data**, applies **portfolio optimization techniques** (Max Sharpe Ratio & Minimum Volatility portfolios), and visualizes the **Efficient Frontier**.  

---

## Features

- **Fetch live market data** using [Yahoo Finance](https://pypi.org/project/yfinance/).  
- **Calculate returns & covariance** matrix of selected assets.  
- **Optimize portfolio** using:  
  - Maximum Sharpe Ratio Portfolio  
  - Minimum Volatility Portfolio  
- **Discrete allocation** of shares for real investment amounts.  
- **Efficient Frontier visualization** with Plotly interactive charts.  
- **Downloadable portfolio report** in CSV format.  
- **Configurable parameters** via sidebar:  
  - Stock tickers  
  - Investment amount  
  - Date range  
  - Risk-free rate  
  - Maximum weight per asset  

---

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Data Source:** [Yahoo Finance](https://pypi.org/project/yfinance/)  
- **Optimization Library:** [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)  
- **Visualization:** [Plotly](https://plotly.com/python/)  
- **Backend:** Python (NumPy, Pandas, Datetime)  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimizer.git
   cd portfolio-optimizer
   
2. Create a virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt

---

