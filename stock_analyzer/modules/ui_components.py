"""
Optimized UI Components Module
Handles all UI component rendering with improved layout and interpretability
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Custom CSS for better styling
def inject_custom_css():
    """Inject custom CSS for better component styling"""
    st.markdown("""
    <style>
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 4px solid #007acc;
    }
    .status-indicator {
        font-size: 1.2em;
        margin-left: 0.5rem;
    }
    .compact-metric {
        text-align: center;
        padding: 0.5rem;
    }
    .health-score {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-excellent { background: linear-gradient(135deg, #d4edda, #c3e6cb); }
    .score-good { background: linear-gradient(135deg, #fff3cd, #ffeaa7); }
    .score-poor { background: linear-gradient(135deg, #f8d7da, #f5c6cb); }
    </style>
    """, unsafe_allow_html=True)

def get_status_indicator(value: float, thresholds: Tuple[float, float], reverse: bool = False) -> Tuple[str, str]:
    """
    Get status indicator emoji and text based on value and thresholds
    
    Args:
        value: Value to evaluate
        thresholds: (good_threshold, excellent_threshold)
        reverse: If True, lower values are better
    
    Returns:
        Tuple of (emoji, status_text)
    """
    if reverse:
        if value <= thresholds[0]:
            return "🟢", "Excellent"
        elif value <= thresholds[1]:
            return "🟡", "Good"
        else:
            return "🔴", "Poor"
    else:
        if value >= thresholds[1]:
            return "🟢", "Excellent"
        elif value >= thresholds[0]:
            return "🟡", "Good"
        else:
            return "🔴", "Poor"

def render_metric_with_status(label: str, value: Any, status_emoji: str = "", 
                            delta: Optional[str] = None, help_text: Optional[str] = None):
    """Render a metric with status indicator in a compact format"""
    if isinstance(value, (int, float)) and value != 'N/A':
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        else:
            display_value = f"{value:,}"
    else:
        display_value = str(value)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label, display_value, delta=delta, help=help_text)
    with col2:
        if status_emoji:
            st.markdown(f"<div class='status-indicator'>{status_emoji}</div>", 
                       unsafe_allow_html=True)

def render_price_range_slider(info: Dict[str, Any], current_price: float):
    """Optimized 52-week price range with better visual hierarchy"""
    st.subheader("📊 52-Week Price Range")
    
    high_52w = info.get('fiftyTwoWeekHigh', current_price)
    low_52w = info.get('fiftyTwoWeekLow', current_price)
    
    if high_52w == 'N/A' or low_52w == 'N/A' or high_52w <= low_52w:
        st.info("52-week range data not available")
        return
    
    # Calculate metrics
    range_position = (current_price - low_52w) / (high_52w - low_52w)
    distance_from_high = (high_52w - current_price) / high_52w * 100
    distance_from_low = (current_price - low_52w) / low_52w * 100
    
    # Position status
    emoji, status = get_status_indicator(range_position, (0.3, 0.7))
    
    # Main slider
    st.slider(
        "Current Position",
        min_value=float(low_52w),
        max_value=float(high_52w),
        value=float(current_price),
        disabled=True,
        help=f"Position: {range_position*100:.1f}% through range"
    )
    
    # Compact metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("52W Low", f"${low_52w:.2f}")
    with col2:
        st.metric("52W High", f"${high_52w:.2f}")
    with col3:
        st.metric("From High", f"-{distance_from_high:.1f}%")
    with col4:
        st.metric("Position", f"{emoji} {status}")

def render_volume_analysis(data: pd.DataFrame, info: Dict[str, Any]):
    """Optimized volume analysis with consolidated metrics"""
    st.subheader("📦 Volume & Liquidity Analysis")
    
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].mean()
    avg_volume_10d = info.get('averageDailyVolume10Day', avg_volume)
    
    # Volume metrics
    volume_vs_avg = ((volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
    volume_emoji, _ = get_status_indicator(abs(volume_vs_avg), (20, 50))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Volume", f"{volume:,.0f}")
    
    with col2:
        render_metric_with_status(
            "vs Average", 
            f"{volume_vs_avg:+.1f}%",
            volume_emoji,
            help_text="Volume compared to historical average"
        )
    
    with col3:
        # Market microstructure
        bid = info.get('bid', 0)
        ask = info.get('ask', 0)
        
        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / current_price * 100) if 'current_price' in locals() else 0
            spread_emoji, _ = get_status_indicator(spread_pct, (0.5, 1.0), reverse=True)
            render_metric_with_status(
                "Bid-Ask Spread", 
                f"{spread_pct:.2f}%",
                spread_emoji,
                help_text="Lower spreads indicate better liquidity"
            )
        else:
            st.metric("Bid-Ask Spread", "N/A")
    
    # Beta analysis
    beta = info.get('beta', 'N/A')
    if beta != 'N/A':
        if beta < 0.8:
            beta_status = "🟢 Low Risk"
        elif beta <= 1.2:
            beta_status = "🟡 Market Risk"
        else:
            beta_status = "🔴 High Risk"
        
        st.metric("Beta (Market Risk)", f"{beta:.2f} - {beta_status}")

def render_financial_health_dashboard(ratios: Dict[str, Any], info: Dict[str, Any]):
    """Consolidated financial health dashboard"""
    if not ratios:
        return
    
    st.subheader("💰 Financial Health Dashboard")
    
    # Calculate overall health score
    health_score = calculate_health_score(ratios, info)
    
    # Health score display
    if health_score >= 80:
        score_class = "score-excellent"
        grade = "A"
        emoji = "🟢"
    elif health_score >= 60:
        score_class = "score-good"
        grade = "B"
        emoji = "🟡"
    else:
        score_class = "score-poor"
        grade = "C"
        emoji = "🔴"
    
    st.markdown(f"""
    <div class="health-score {score_class}">
        <h3>{emoji} Financial Health Score: {health_score}/100 (Grade {grade})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📈 Profitability", "⚖️ Financial Strength", "📊 Valuation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_profitability_metrics(ratios)
        with col2:
            render_efficiency_metrics(ratios)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            render_liquidity_metrics(ratios)
        with col2:
            render_leverage_metrics(ratios)
    
    with tab3:
        render_valuation_metrics(ratios, info)

def render_profitability_metrics(ratios: Dict[str, Any]):
    """Render profitability metrics with status indicators"""
    st.markdown("**Profitability Ratios**")
    
    # Gross Margin
    gross_margin = ratios.get('Gross Margin %', 'N/A')
    if gross_margin != 'N/A':
        emoji, _ = get_status_indicator(gross_margin, (20, 40))
        render_metric_with_status("Gross Margin", f"{gross_margin:.1f}%", emoji)
    
    # Net Margin
    net_margin = ratios.get('Net Margin %', 'N/A')
    if net_margin != 'N/A':
        emoji, _ = get_status_indicator(net_margin, (5, 15))
        render_metric_with_status("Net Margin", f"{net_margin:.1f}%", emoji)

def render_efficiency_metrics(ratios: Dict[str, Any]):
    """Render efficiency metrics"""
    st.markdown("**Efficiency Ratios**")
    
    # ROA
    roa = ratios.get('ROA %', 'N/A')
    if roa != 'N/A':
        emoji, _ = get_status_indicator(roa, (5, 15))
        render_metric_with_status("ROA", f"{roa:.1f}%", emoji)
    
    # ROE
    roe = ratios.get('ROE %', 'N/A')
    if roe != 'N/A':
        emoji, _ = get_status_indicator(roe, (10, 20))
        render_metric_with_status("ROE", f"{roe:.1f}%", emoji)

def render_liquidity_metrics(ratios: Dict[str, Any]):
    """Render liquidity metrics"""
    st.markdown("**Liquidity Ratios**")
    
    current_ratio = ratios.get('Current Ratio', 'N/A')
    if current_ratio != 'N/A':
        emoji, status = get_status_indicator(current_ratio, (1.0, 2.0))
        render_metric_with_status("Current Ratio", f"{current_ratio:.2f}", emoji)
        st.caption(f"Liquidity: {status}")

def render_leverage_metrics(ratios: Dict[str, Any]):
    """Render leverage metrics"""
    st.markdown("**Leverage Ratios**")
    
    debt_equity = ratios.get('Debt to Equity', 'N/A')
    if debt_equity != 'N/A':
        emoji, status = get_status_indicator(debt_equity, (1.0, 2.0), reverse=True)
        render_metric_with_status("Debt/Equity", f"{debt_equity:.2f}", emoji)
        st.caption(f"Leverage: {status}")

def render_valuation_metrics(ratios: Dict[str, Any], info: Dict[str, Any]):
    """Render valuation metrics"""
    st.markdown("**Valuation Ratios**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pe_ratio = info.get('trailingPE', 'N/A')
        if pe_ratio != 'N/A':
            # PE interpretation depends on industry, but general guidelines
            if 10 <= pe_ratio <= 25:
                emoji = "🟢"
            elif 5 <= pe_ratio <= 35:
                emoji = "🟡"
            else:
                emoji = "🔴"
            render_metric_with_status("P/E Ratio", f"{pe_ratio:.1f}", emoji)
        
        pb_ratio = ratios.get('P/B Ratio', 'N/A')
        if pb_ratio != 'N/A':
            emoji, _ = get_status_indicator(pb_ratio, (3.0, 5.0), reverse=True)
            render_metric_with_status("P/B Ratio", f"{pb_ratio:.2f}", emoji)
    
    with col2:
        ps_ratio = ratios.get('P/S Ratio', 'N/A')
        if ps_ratio != 'N/A':
            emoji, _ = get_status_indicator(ps_ratio, (5.0, 10.0), reverse=True)
            render_metric_with_status("P/S Ratio", f"{ps_ratio:.2f}", emoji)

def render_short_interest_analysis(info: Dict[str, Any]):
    """Optimized short interest analysis"""
    st.subheader("🩳 Short Interest Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        short_ratio = info.get('shortRatio', 'N/A')
        if short_ratio != 'N/A':
            emoji, _ = get_status_indicator(short_ratio, (3, 7))
            render_metric_with_status(
                "Days to Cover", 
                f"{short_ratio:.1f}",
                emoji,
                help_text="Days to cover all short positions"
            )
    
    with col2:
        short_percent = info.get('shortPercentOfFloat', 'N/A')
        if short_percent != 'N/A':
            short_pct = short_percent * 100
            emoji, status = get_status_indicator(short_pct, (10, 20))
            render_metric_with_status(
                "Short % of Float", 
                f"{short_pct:.1f}%",
                emoji
            )
            st.caption(f"Sentiment: {status}")

def calculate_health_score(ratios: Dict[str, Any], info: Dict[str, Any]) -> int:
    """Calculate overall financial health score (0-100)"""
    score = 0
    
    # Profitability (25 points)
    roe = ratios.get('ROE %', 0)
    if roe > 20: score += 15
    elif roe > 10: score += 10
    elif roe > 5: score += 5
    
    net_margin = ratios.get('Net Margin %', 0)
    if net_margin > 15: score += 10
    elif net_margin > 5: score += 5
    
    # Financial Strength (25 points)
    current_ratio = ratios.get('Current Ratio', 0)
    if current_ratio > 2: score += 15
    elif current_ratio > 1: score += 10
    elif current_ratio > 0.5: score += 5
    
    debt_equity = ratios.get('Debt to Equity', 0)
    if debt_equity < 0.5: score += 10
    elif debt_equity < 1: score += 5
    
    # Market Performance (25 points)
    pe_ratio = info.get('trailingPE', 0)
    if 10 < pe_ratio < 25: score += 15
    elif 5 < pe_ratio < 35: score += 10
    
    dividend_yield = info.get('dividendYield', 0)
    if dividend_yield and dividend_yield > 0.03: score += 10
    elif dividend_yield and dividend_yield > 0.01: score += 5
    
    # Growth & Efficiency (25 points)
    roa = ratios.get('ROA %', 0)
    if roa > 15: score += 15
    elif roa > 5: score += 10
    elif roa > 2: score += 5
    
    # Market cap stability bonus
    market_cap = info.get('marketCap', 0)
    if market_cap > 10e9: score += 10  # Large cap bonus
    
    return min(score, 100)

def render_compact_sidebar_stats(data: pd.DataFrame, info: Dict[str, Any]):
    """Render compact sidebar statistics"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    
    # Price change
    if len(data) >= 2:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price * 100)
        
        emoji = "🟢" if change_pct > 0 else "🔴"
        st.sidebar.metric("Daily Change", f"{change_pct:+.2f}%", 
                         delta=f"${current_price - prev_price:+.2f}")
    
    # Volume status
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].mean()
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        volume_status = "🔥 High Volume"
    elif volume_ratio > 0.8:
        volume_status = "📊 Normal Volume"
    else:
        volume_status = "📉 Low Volume"
    
    st.sidebar.write(volume_status)
    
    # Market cap category
    market_cap = info.get('marketCap', 0)
    if market_cap > 200e9:
        cap_category = "🏢 Mega Cap ($200B+)"
    elif market_cap > 10e9:
        cap_category = "🏬 Large Cap"
    elif market_cap > 2e9:
        cap_category = "🏪 Mid Cap"
    else:
        cap_category = "🏫 Small Cap"
    
    st.sidebar.write(f"**Size**: {cap_category}")
