"""
Optimized Technical Analysis Module
Handles all technical indicators and chart creation using established libraries
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Try to import technical analysis libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def calculate_technical_indicators(data):
    """
    Calculate technical indicators using optimized libraries
    """
    df = data.copy()
    
    if TALIB_AVAILABLE:
        # Use TA-Lib (fastest and most accurate)
        df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'].values, timeperiod=50)
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
            df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Additional indicators
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
        
    elif PANDAS_TA_AVAILABLE:
        # Use Pandas-TA as fallback
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        
        # Rename columns to match expected format
        column_mapping = {
            'SMA_20': 'SMA_20',
            'SMA_50': 'SMA_50',
            'RSI_14': 'RSI',
            'BBL_20_2.0': 'BB_Lower',
            'BBM_20_2.0': 'BB_Middle',
            'BBU_20_2.0': 'BB_Upper',
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'MACD_Signal',
            'MACDh_12_26_9': 'MACD_Hist',
            'ATRr_14': 'ATR'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
    else:
        # Fallback to optimized custom implementations
        df = _calculate_indicators_custom(df)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['Price_Change'] = df['Close'].pct_change()
    
    return df

def _calculate_indicators_custom(df):
    """Optimized custom implementations as fallback"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Moving averages
    df['SMA_20'] = close.rolling(20, min_periods=1).mean()
    df['SMA_50'] = close.rolling(50, min_periods=1).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']
    bb_std = close.rolling(20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI (optimized)
    df['RSI'] = _calculate_rsi_optimized(close)
    
    # MACD (optimized)
    df['MACD'], df['MACD_Signal'] = _calculate_macd_optimized(close)
    
    # ATR
    df['ATR'] = _calculate_atr_optimized(high, low, close)
    
    return df

def _calculate_rsi_optimized(prices, window=14):
    """Optimized RSI calculation"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_macd_optimized(prices, fast=12, slow=26, signal=9):
    """Optimized MACD calculation"""
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    return macd, signal_line

def _calculate_atr_optimized(high, low, close, window=14):
    """Optimized ATR calculation"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=window, min_periods=window).mean()
    return atr

@st.cache_data(ttl=300)
def create_candlestick_chart(data, ticker, show_sma=False, show_bollinger=False, show_volume=True):
    """
    Create optimized candlestick chart with technical indicators
    """
    indicators = {
        'sma': show_sma,
        'bollinger': show_bollinger,
        'volume': show_volume
    }
    
    # Determine subplot configuration
    rows = 1
    subplot_titles = [f'{ticker} Stock Price']
    row_heights = [1.0]
    
    if show_volume:
        rows += 1
        subplot_titles.append('Volume')
        row_heights = [0.7, 0.3]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add indicators
    _add_price_indicators(fig, data, indicators)
    
    # Add volume if requested
    if show_volume:
        _add_volume_chart(fig, data, rows)
    
    # Optimize layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        template='plotly_white',
        height=600 if show_volume else 400,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def _add_price_indicators(fig, data, indicators):
    """Add price-based indicators to the chart"""
    if indicators.get('sma', False):
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['SMA_20'], 
                    name='SMA 20', line=dict(color='orange', width=1.5),
                    opacity=0.8
                ), row=1, col=1
            )
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['SMA_50'], 
                    name='SMA 50', line=dict(color='red', width=1.5),
                    opacity=0.8
                ), row=1, col=1
            )
    
    if indicators.get('bollinger', False):
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Upper'], 
                    name='BB Upper', line=dict(color='gray', width=1),
                    showlegend=False, opacity=0.5
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Lower'], 
                    name='BB Lower', line=dict(color='gray', width=1),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False, opacity=0.5
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Middle'], 
                    name='BB Middle', line=dict(color='blue', width=1, dash='dot'),
                    opacity=0.7
                ), row=1, col=1
            )

def _add_volume_chart(fig, data, row):
    """Add volume chart efficiently"""
    colors = np.where(data['Close'] >= data['Open'], '#26a69a', '#ef5350')
    
    fig.add_trace(
        go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume', marker_color=colors,
            opacity=0.7, showlegend=False
        ), row=row, col=1
    )

def create_rsi_chart(data, ticker):
    """Create RSI indicator chart"""
    fig = go.Figure()
    
    if 'RSI' in data.columns:
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
    """Create MACD indicator chart"""
    fig = go.Figure()
    
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
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
        
        # Histogram
        if 'MACD_Hist' in data.columns:
            histogram = data['MACD_Hist']
        else:
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

def get_technical_signals(data):
    """
    Generate comprehensive technical signals with scoring
    """
    signals = {}
    total_score = 0
    max_score = 0
    
    # RSI Signal (Weight: 2)
    if 'RSI' in data.columns:
        rsi = data['RSI'].iloc[-1]
        if pd.notna(rsi):
            if rsi > 70:
                signals['RSI'] = 'Overbought - Consider Sell'
                total_score -= 2
            elif rsi < 30:
                signals['RSI'] = 'Oversold - Consider Buy'
                total_score += 2
            else:
                signals['RSI'] = 'Neutral'
            max_score += 2
    
    # MACD Signal (Weight: 2)
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal:
                signals['MACD'] = 'Bullish - MACD above signal'
                total_score += 2
            else:
                signals['MACD'] = 'Bearish - MACD below signal'
                total_score -= 2
            max_score += 2
    
    # Moving Average Signal (Weight: 1)
    if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
        price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        if pd.notna(sma_20) and pd.notna(sma_50):
            if price > sma_20 > sma_50:
                signals['Moving_Average'] = 'Bullish - Price above both MAs'
                total_score += 1
            elif price < sma_20 < sma_50:
                signals['Moving_Average'] = 'Bearish - Price below both MAs'
                total_score -= 1
            else:
                signals['Moving_Average'] = 'Mixed signals'
            max_score += 1
    
    # Bollinger Bands Signal
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
        price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            if price > bb_upper:
                signals['Bollinger_Bands'] = 'Overbought - Price above upper band'
            elif price < bb_lower:
                signals['Bollinger_Bands'] = 'Oversold - Price below lower band'
            else:
                signals['Bollinger_Bands'] = 'Normal range'
    
    # Overall sentiment
    if max_score > 0:
        sentiment_score = total_score / max_score
        if sentiment_score > 0.3:
            signals['Overall_Sentiment'] = 'Bullish'
        elif sentiment_score < -0.3:
            signals['Overall_Sentiment'] = 'Bearish'
        else:
            signals['Overall_Sentiment'] = 'Neutral'
    
    return signals

# Keep some of your original functions that are still useful
def get_support_resistance_levels(data, window=20):
    """Calculate support and resistance levels"""
    highs = data['High'].rolling(window=window).max()
    lows = data['Low'].rolling(window=window).min()
    
    recent_high = highs.iloc[-1]
    recent_low = lows.iloc[-1]
    price_range = recent_high - recent_low
    
    levels = {
        'strong_resistance': recent_high,
        'resistance': recent_high - (price_range * 0.25),
        'support': recent_low + (price_range * 0.25),
        'strong_support': recent_low
    }
    
    return levels

def calculate_volatility_metrics(data):
    """Calculate various volatility metrics"""
    returns = data['Close'].pct_change().dropna()
    
    metrics = {
        'daily_volatility': returns.std(),
        'annualized_volatility': returns.std() * np.sqrt(252),
        'average_true_range': data['ATR'].iloc[-1] if 'ATR' in data.columns else None
    }
    
    return metrics