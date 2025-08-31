"""
UI Components Module
Handles all UI component rendering and layout
"""

import streamlit as st
import pandas as pd

def render_price_range_slider(info, current_price):
    """
    Render 52-week price range slider
    
    Args:
        info (dict): Stock info from yfinance
        current_price (float): Current stock price
    """
    st.markdown("---")
    st.subheader("📊 52-Week Price Range")
    
    high_52w = info.get('fiftyTwoWeekHigh', current_price)
    low_52w = info.get('fiftyTwoWeekLow', current_price)
    
    if high_52w != 'N/A' and low_52w != 'N/A' and high_52w > low_52w:
        # Calculate position within range
        range_position = (current_price - low_52w) / (high_52w - low_52w)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a visual slider representation
            st.slider(
                label="Current Position in 52W Range",
                min_value=float(low_52w),
                max_value=float(high_52w),
                value=float(current_price),
                disabled=True,
                help=f"Current price ${current_price:.2f} is {range_position*100:.1f}% through the 52-week range"
            )
            
            # Add range labels
            col_low, col_mid, col_high = st.columns(3)
            with col_low:
                st.caption(f"52W Low: ${low_52w:.2f}")
            with col_mid:
                # Color code based on position
                if range_position > 0.8:
                    position_color = "🟢"
                    position_text = "Near High"
                elif range_position < 0.2:
                    position_color = "🔴"
                    position_text = "Near Low"
                else:
                    position_color = "🟡"
                    position_text = "Mid Range"
                st.caption(f"{position_color} {position_text}")
            with col_high:
                st.caption(f"52W High: ${high_52w:.2f}")
        
        with col2:
            # Distance from high/low
            distance_from_high = ((high_52w - current_price) / high_52w * 100)
            distance_from_low = ((current_price - low_52w) / low_52w * 100)
            st.metric("Distance from High", f"-{distance_from_high:.1f}%")
            st.metric("Distance from Low", f"+{distance_from_low:.1f}%")

def render_volume_analysis(data, info):
    """
    Render volume analysis section
    
    Args:
        data (pd.DataFrame): Stock OHLCV data
        info (dict): Stock info from yfinance
    """
    st.markdown("---")
    st.subheader("📦 Volume Analysis")
    
    volume_col1, volume_col2 = st.columns(2)
    
    with volume_col1:
        with st.container():
            st.markdown("**Trading Volume**")
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].mean()
            avg_volume_10d = info.get('averageDailyVolume10Day', 'N/A')
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("Current Volume", f"{volume:,.0f}")
            with col_v2:
                st.metric(
                    "vs Average", 
                    f"{((volume - avg_volume) / avg_volume * 100):+.1f}%",
                    delta=f"{volume - avg_volume:,.0f}"
                )
            
            if avg_volume_10d != 'N/A':
                st.metric("10-Day Avg", f"{avg_volume_10d:,.0f}")
    
    with volume_col2:
        with st.container():
            st.markdown("**Market Microstructure**")
            bid = info.get('bid', 0)
            ask = info.get('ask', 0)
            beta = info.get('beta', 'N/A')
            
            if bid > 0 and ask > 0:
                spread = ask - bid
                spread_pct = (spread / data['Close'].iloc[-1] * 100)
                st.metric("Bid-Ask Spread", f"${spread:.2f} ({spread_pct:.2f}%)")
            else:
                st.metric("Bid-Ask Spread", "N/A")
            
            # Beta with interpretation
            if beta != 'N/A':
                if beta < 1:
                    beta_interpretation = "Less Volatile"
                    beta_color = "🟢"
                elif beta > 1.5:
                    beta_interpretation = "Highly Volatile"
                    beta_color = "🔴"
                else:
                    beta_interpretation = "More Volatile"
                    beta_color = "🟡"
                st.metric("Beta", f"{beta:.2f} {beta_color}", help=f"{beta_interpretation} than market")
            else:
                st.metric("Beta", "N/A")

def render_short_interest_analysis(info):
    """
    Render short interest analysis section
    
    Args:
        info (dict): Stock info from yfinance
    """
    st.markdown("---")
    st.subheader("🩳 Short Interest Analysis")
    
    short_col1, short_col2 = st.columns(2)
    
    with short_col1:
        short_ratio = info.get('shortRatio', 'N/A')
        st.metric(
            "Short Ratio (Days to Cover)", 
            f"{short_ratio:.2f}" if short_ratio != 'N/A' else "N/A",
            help="Number of days to cover all short positions based on average volume"
        )
    
    with short_col2:
        short_percent = info.get('shortPercentOfFloat', 'N/A')
        if short_percent != 'N/A':
            short_pct_display = short_percent * 100
            # Add interpretation
            if short_pct_display > 20:
                short_sentiment = "🔴 High Short Interest"
            elif short_pct_display > 10:
                short_sentiment = "🟡 Moderate Short Interest"
            else:
                short_sentiment = "🟢 Low Short Interest"
            st.metric(
                "Short % of Float", 
                f"{short_pct_display:.2f}%",
                help=short_sentiment
            )
        else:
            st.metric("Short % of Float", "N/A")

def render_financial_ratios(ratios):
    """
    Render financial ratios section
    
    Args:
        ratios (dict): Dictionary of calculated financial ratios
    """
    if not ratios:
        return
        
    st.markdown("---")
    st.subheader("💰 Key Financial Ratios")
    
    # Profitability Box
    profit_col1, profit_col2 = st.columns(2)
    
    with profit_col1:
        st.markdown("**📈 Profitability Ratios**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            gross_margin = ratios.get('Gross Margin %', 'N/A')
            st.metric("Gross Margin", f"{gross_margin:.2f}%" if gross_margin != 'N/A' else "N/A")
        with p_col2:
            net_margin = ratios.get('Net Margin %', 'N/A')
            st.metric("Net Margin", f"{net_margin:.2f}%" if net_margin != 'N/A' else "N/A")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            roa = ratios.get('ROA %', 'N/A')
            st.metric("ROA", f"{roa:.2f}%" if roa != 'N/A' else "N/A")
        with r_col2:
            roe = ratios.get('ROE %', 'N/A')
            st.metric("ROE", f"{roe:.2f}%" if roe != 'N/A' else "N/A")
    
    with profit_col2:
        st.markdown("**⚖️ Financial Health Ratios**")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            current_ratio = ratios.get('Current Ratio', 'N/A')
            # Add health indicator
            if current_ratio != 'N/A':
                if current_ratio > 2:
                    health_indicator = "🟢"
                elif current_ratio > 1:
                    health_indicator = "🟡"
                else:
                    health_indicator = "🔴"
                st.metric("Current Ratio", f"{current_ratio:.2f} {health_indicator}")
            else:
                st.metric("Current Ratio", "N/A")
        
        with h_col2:
            debt_equity = ratios.get('Debt to Equity', 'N/A')
            # Add leverage indicator
            if debt_equity != 'N/A':
                if debt_equity > 2:
                    leverage_indicator = "🔴"
                elif debt_equity > 1:
                    leverage_indicator = "🟡"
                else:
                    leverage_indicator = "🟢"
                st.metric("Debt/Equity", f"{debt_equity:.2f} {leverage_indicator}")
            else:
                st.metric("Debt/Equity", "N/A")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            pb_ratio = ratios.get('P/B Ratio', 'N/A')
            st.metric("P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio != 'N/A' else "N/A")
        with v_col2:
            ps_ratio = ratios.get('P/S Ratio', 'N/A')
            st.metric("P/S Ratio", f"{ps_ratio:.2f}" if ps_ratio != 'N/A' else "N/A")

def render_metric_card(title, value, delta=None, help_text=None):
    """
    Render a custom metric card
    
    Args:
        title (str): Metric title
        value (str): Metric value
        delta (str, optional): Delta value
        help_text (str, optional): Help tooltip text
    """
    with st.container():
        st.markdown(f"""
        <div class="metric-container">
            <h4>{title}</h4>
            <h2>{value}</h2>
            {f'<p style="color: green;">{delta}</p>' if delta else ''}
            {f'<small>{help_text}</small>' if help_text else ''}
        </div>
        """, unsafe_allow_html=True)

def render_data_quality_indicator(data_quality_score):
    """
    Render data quality indicator
    
    Args:
        data_quality_score (float): Score from 0-100
    """
    if data_quality_score >= 90:
        color = "🟢"
        status = "Excellent"
    elif data_quality_score >= 70:
        color = "🟡"
        status = "Good"
    else:
        color = "🔴"
        status = "Limited"
    
    st.sidebar.markdown(f"**Data Quality**: {color} {status} ({data_quality_score:.0f}%)")

def render_loading_placeholder():
    """Render loading placeholder"""
    with st.spinner("Loading financial data..."):
        # Create placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.empty()
        with col2:
            st.empty()
        with col3:
            st.empty()
        with col4:
            st.empty()

def render_error_message(error_type, message):
    """
    Render consistent error messages
    
    Args:
        error_type (str): Type of error (warning, error, info)
        message (str): Error message
    """
    if error_type == "warning":
        st.warning(f"⚠️ {message}")
    elif error_type == "error":
        st.error(f"❌ {message}")
    elif error_type == "info":
        st.info(f"ℹ️ {message}")
    else:
        st.write(message)

def render_financial_health_score(ratios, info):
    """
    Render overall financial health score
    
    Args:
        ratios (dict): Financial ratios
        info (dict): Stock info
    """
    try:
        score = 0
        max_score = 100
        
        # Profitability (30 points)
        roe = ratios.get('ROE %', 0)
        if roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe > 5:
            score += 5
        
        net_margin = ratios.get('Net Margin %', 0)
        if net_margin > 20:
            score += 15
        elif net_margin > 10:
            score += 10
        elif net_margin > 5:
            score += 5
        
        # Liquidity (20 points)
        current_ratio = ratios.get('Current Ratio', 0)
        if current_ratio > 2:
            score += 20
        elif current_ratio > 1:
            score += 15
        elif current_ratio > 0.5:
            score += 10
        
        # Leverage (20 points)
        debt_equity = ratios.get('Debt to Equity', 0)
        if debt_equity < 0.5:
            score += 20
        elif debt_equity < 1:
            score += 15
        elif debt_equity < 2:
            score += 10
        
        # Market metrics (30 points)
        pe_ratio = info.get('trailingPE', 0)
        if 10 < pe_ratio < 25:
            score += 15
        elif 5 < pe_ratio < 35:
            score += 10
        
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield and dividend_yield > 0.02:
            score += 15
        elif dividend_yield and dividend_yield > 0.01:
            score += 10
        
        # Render score
        if score >= 80:
            color = "🟢"
            grade = "A"
        elif score >= 70:
            color = "🟡"
            grade = "B"
        elif score >= 60:
            color = "🟡"
            grade = "C"
        else:
            color = "🔴"
            grade = "D"
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Financial Health Score**")
        st.sidebar.markdown(f"## {color} {score}/100 ({grade})")
        
        # Score breakdown
        with st.sidebar.expander("Score Breakdown"):
            st.write("**Profitability**: 30 pts")
            st.write("**Liquidity**: 20 pts") 
            st.write("**Leverage**: 20 pts")
            st.write("**Market Metrics**: 30 pts")
        
    except Exception as e:
        pass

def render_quick_stats_sidebar(data, info):
    """
    Render quick stats in sidebar
    
    Args:
        data (pd.DataFrame): Stock data
        info (dict): Stock info
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Stats**")
    
    # Price change
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price * 100)
    
    color = "🟢" if change_pct > 0 else "🔴"
    st.sidebar.metric("Today's Change", f"{change_pct:+.2f}%", label_visibility="collapsed")
    
    # Volume vs average
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].mean()
    volume_ratio = volume / avg_volume
    
    if volume_ratio > 1.5:
        volume_status = "🔥 High"
    elif volume_ratio > 0.8:
        volume_status = "📊 Normal"
    else:
        volume_status = "📉 Low"
    
    st.sidebar.write(f"**Volume**: {volume_status}")
    
    # Market cap category
    market_cap = info.get('marketCap', 0)
    if market_cap > 200e9:
        cap_category = "🏢 Mega Cap"
    elif market_cap > 10e9:
        cap_category = "🏬 Large Cap"
    elif market_cap > 2e9:
        cap_category = "🏪 Mid Cap"
    else:
        cap_category = "🏫 Small Cap"
    
    st.sidebar.write(f"**Size**: {cap_category}")
