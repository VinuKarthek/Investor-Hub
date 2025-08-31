"""
Data Fetching Module
Handles all data retrieval from Yahoo Finance
"""

import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(ticker, period="1y"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for data
        
    Returns:
        tuple: (stock_data, stock_info) or (None, None) if error
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_financial_statements(ticker):
    """
    Fetch financial statements from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (income_stmt, balance_sheet, cash_flow) or (None, None, None) if error
    """
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        return income_stmt, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching financial statements for {ticker}: {str(e)}")
        return None, None, None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_analyst_recommendations(ticker):
    """
    Get analyst recommendations and target prices
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (recommendations, analyst_price_targets) or (None, None) if error
    """
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        analyst_price_targets = stock.analyst_price_targets
        return recommendations, analyst_price_targets
    except Exception as e:
        st.warning(f"Analyst data not available for {ticker}")
        return None, None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_dividend_data(ticker):
    """
    Get dividend history data
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        pd.Series: Dividend data or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        return dividends if not dividends.empty else None
    except Exception as e:
        return None

def validate_ticker(ticker):
    """
    Validate if ticker exists and has basic data
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if basic info exists
        return 'symbol' in info or 'shortName' in info
    except:
        return False
