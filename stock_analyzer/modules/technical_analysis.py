"""
Technical Analysis Module
Handles all technical indicators and chart creation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for stock data
    
    Args:
        data (pd.DataFrame): Stock OHLCV data
        
    Returns:
        pd.DataFrame: Data with technical indicators added
    """
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # RSI (Relative Strength Index)
    data['RSI'] = calculate_rsi(data['Close'])
    
    # MACD
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    
    # Price change
    data['Price_Change'] = data['Close'].pct_change()
    
    return data

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index
    
    Args:
        prices (pd.Series): Price series
        window (int): RSI calculation window
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line EMA period
        
    Returns:
        tuple: (MACD line, Signal line)
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def create_candlestick_chart(data, ticker, show_sma=False, show_bollinger=False, show_volume=True):
    """
    Create interactive candlestick chart with technical indicators
    
    Args:
        data (pd.DataFrame): Stock OHLCV data with indicators
        ticker (str): Stock ticker symbol
        show_sma (bool): Whether to show moving averages
        show_bollinger (bool): Whether to show Bollinger Bands
        show_volume (bool): Whether to show volume chart
        
    Returns:
        plotly.graph_objects.Figure: Interactive candlestick chart
    """
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Stock Price', 'Volume') if show_volume else (f'{ticker} Stock Price',),
        row_heights=[0.7, 0.3] if show_volume else [1.0]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # Add technical indicators if selected
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['SMA_20'], 
                name='SMA 20',
                line=dict(color='orange', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['SMA_50'], 
                name='SMA 50',
                line=dict(color='red', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['BB_Upper'], 
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                opacity=0.5
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['BB_Lower'], 
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False,
                opacity=0.5
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['BB_Middle'], 
                name='BB Middle',
                line=dict(color='blue', width=1, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Volume chart
    if show_volume:
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Stock Price (USD)',
        template='plotly_white',
        height=700 if show_volume else 500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Remove rangeslider
    fig.update_xaxes(rangeslider_visible=False)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_rsi_chart(data, ticker):
    """
    Create RSI indicator chart
    
    Args:
        data (pd.DataFrame): Stock data with RSI
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: RSI chart
    """
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    fig.update_layout(
        title=f'{ticker} RSI (14)',
        xaxis_title='Date',
        yaxis_title='RSI',
        height=300,
        template='plotly_white',
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_macd_chart(data, ticker):
    """
    Create MACD indicator chart
    
    Args:
        data (pd.DataFrame): Stock data with MACD
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: MACD chart
    """
    fig = go.Figure()
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD_Signal'],
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # Histogram (MACD - Signal)
    histogram = data['MACD'] - data['MACD_Signal']
    colors = ['green' if x >= 0 else 'red' for x in histogram]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=histogram,
        name='Histogram',
        marker_color=colors,
        opacity=0.6
    ))
    
    fig.update_layout(
        title=f'{ticker} MACD',
        xaxis_title='Date',
        yaxis_title='MACD',
        height=300,
        template='plotly_white'
    )
    
    return fig

def get_support_resistance_levels(data, window=20):
    """
    Calculate support and resistance levels
    
    Args:
        data (pd.DataFrame): Stock OHLC data
        window (int): Window for calculating levels
        
    Returns:
        dict: Support and resistance levels
    """
    highs = data['High'].rolling(window=window).max()
    lows = data['Low'].rolling(window=window).min()
    
    # Get recent levels
    recent_high = highs.iloc[-1]
    recent_low = lows.iloc[-1]
    
    # Calculate additional levels based on price action
    price_range = recent_high - recent_low
    
    levels = {
        'strong_resistance': recent_high,
        'resistance': recent_high - (price_range * 0.25),
        'support': recent_low + (price_range * 0.25),
        'strong_support': recent_low
    }
    
    return levels

def calculate_volatility_metrics(data):
    """
    Calculate various volatility metrics
    
    Args:
        data (pd.DataFrame): Stock price data
        
    Returns:
        dict: Volatility metrics
    """
    returns = data['Close'].pct_change().dropna()
    
    metrics = {
        'daily_volatility': returns.std(),
        'annualized_volatility': returns.std() * np.sqrt(252),
        'average_true_range': calculate_atr(data),
        'volatility_percentile': None  # Would need longer history
    }
    
    return metrics

def calculate_atr(data, window=14):
    """
    Calculate Average True Range
    
    Args:
        data (pd.DataFrame): Stock OHLC data
        window (int): ATR calculation window
        
    Returns:
        pd.Series: ATR values
    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_stochastic(data, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator
    
    Args:
        data (pd.DataFrame): Stock OHLC data
        k_window (int): %K calculation window
        d_window (int): %D smoothing window
        
    Returns:
        tuple: (%K, %D)
    """
    lowest_low = data['Low'].rolling(window=k_window).min()
    highest_high = data['High'].rolling(window=k_window).max()
    
    k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent, d_percent

def create_volume_profile_chart(data, ticker):
    """
    Create volume profile chart
    
    Args:
        data (pd.DataFrame): Stock OHLCV data
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Volume profile chart
    """
    # Simple volume profile approximation
    price_bins = 50
    price_min = data['Low'].min()
    price_max = data['High'].max()
    
    bins = np.linspace(price_min, price_max, price_bins)
    volume_profile = np.zeros(len(bins) - 1)
    
    for i, (idx, row) in enumerate(data.iterrows()):
        # Find which bin this price falls into
        bin_idx = np.digitize(row['Close'], bins) - 1
        if 0 <= bin_idx < len(volume_profile):
            volume_profile[bin_idx] += row['Volume']
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=bin_centers,
        x=volume_profile,
        orientation='h',
        name='Volume Profile',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'{ticker} Volume Profile',
        xaxis_title='Volume',
        yaxis_title='Price',
        height=500,
        template='plotly_white'
    )
    
    return fig

def calculate_momentum_indicators(data):
    """
    Calculate momentum indicators
    
    Args:
        data (pd.DataFrame): Stock OHLC data
        
    Returns:
        pd.DataFrame: Data with momentum indicators
    """
    # Rate of Change (ROC)
    data['ROC'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
    
    # Williams %R
    high_14 = data['High'].rolling(window=14).max()
    low_14 = data['Low'].rolling(window=14).min()
    data['Williams_R'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
    
    # Commodity Channel Index (CCI)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    return data

def get_technical_signals(data):
    """
    Generate technical trading signals
    
    Args:
        data (pd.DataFrame): Stock data with indicators
        
    Returns:
        dict: Trading signals
    """
    signals = {}
    
    # RSI signals
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi > 70:
        signals['RSI'] = 'Overbought - Consider Sell'
    elif latest_rsi < 30:
        signals['RSI'] = 'Oversold - Consider Buy'
    else:
        signals['RSI'] = 'Neutral'
    
    # Moving Average signals
    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            signals['Moving_Average'] = 'Bullish - Price above both MAs'
        elif current_price < sma_20 < sma_50:
            signals['Moving_Average'] = 'Bearish - Price below both MAs'
        else:
            signals['Moving_Average'] = 'Mixed signals'
    
    # Bollinger Bands signals
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        
        if current_price > bb_upper:
            signals['Bollinger_Bands'] = 'Overbought - Price above upper band'
        elif current_price < bb_lower:
            signals['Bollinger_Bands'] = 'Oversold - Price below lower band'
        else:
            signals['Bollinger_Bands'] = 'Normal range'
    
    # MACD signals
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        
        if macd > macd_signal:
            signals['MACD'] = 'Bullish - MACD above signal'
        else:
            signals['MACD'] = 'Bearish - MACD below signal'
    
    return signal