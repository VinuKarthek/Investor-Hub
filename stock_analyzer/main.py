"""
Stock Financial Dashboard - Main Application (Optimized)
Run this file to start the dashboard: streamlit run main.py
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.data_fetcher import (
    get_stock_data, get_financial_statements, 
    get_analyst_recommendations
)
from modules.technical_analysis import (
    calculate_technical_indicators, 
    create_candlestick_chart,
    create_rsi_chart,
    create_macd_chart,
    get_technical_signals,
    get_support_resistance_levels
)
from modules.financial_charts import (
    create_combined_income_chart, 
    create_combined_balance_sheet_chart,
    create_historical_performance_chart,
    create_historical_ratios_chart,
    create_waterfall_chart,
    create_sankey_diagram
)
from modules.analytics import (
    calculate_financial_ratios,
    create_eps_trend_chart,
    create_dividend_history_chart,
    display_analyst_ratings,
    get_industry_comparison,
    create_industry_comparison_chart
)
from modules.ui_components import (
    render_price_range_slider,
    render_volume_analysis,
    render_short_interest_analysis,
    render_financial_health_dashboard,
    inject_custom_css  # Add this if you implemented the CSS function
)

# Page configuration
st.set_page_config(
    page_title="Stock Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for better styling
def setup_page_styling():
    """Setup custom CSS and page styling"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .tab-content {
            padding: 1rem 0;
        }
        .metric-row {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    setup_page_styling()
    
    st.markdown('<h1 class="main-header">📊 Stock Financial Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar Controls
    render_sidebar()
    
    # Main Dashboard Content
    if 'stock_data' in st.session_state:
        render_tabbed_dashboard()
    else:
        render_welcome_screen()

def render_sidebar():
    """Optimized sidebar with better organization"""
    st.sidebar.header("🎛️ Dashboard Controls")
    
    with st.sidebar.form("stock_form"):
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker", 
            value="AAPL", 
            help="e.g., AAPL, GOOGL, MSFT, TSLA"
        )
        
        # Time period selection
        period_options = {
            "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", 
            "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", 
            "10 Years": "10y", "Year to Date": "ytd", "Max": "max"
        }
        period_label = st.selectbox("Time Period", list(period_options.keys()), index=3)
        period = period_options[period_label]
        
        # Technical indicators
        st.subheader("📈 Technical Indicators")
        show_sma = st.checkbox("Moving Averages (SMA 20, 50)", value=True)
        show_bollinger = st.checkbox("Bollinger Bands")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Fetch Data", type="primary")
        
        if submitted and ticker:
            fetch_all_data(ticker.upper(), period, show_sma, show_bollinger)
    
    # Display current data info if available
    if 'ticker' in st.session_state:
        render_sidebar_info()

def render_sidebar_info():
    """Display current stock info in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 {st.session_state['ticker']}")
    
    if 'stock_data' in st.session_state:
        data = st.session_state['stock_data']
        info = st.session_state['stock_info']
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price * 100) if prev_price != 0 else 0
        
        # Quick metrics
        st.sidebar.metric("Current Price", f"${current_price:.2f}", 
                         delta=f"{change_pct:+.2f}%")
        
        # Market cap
        market_cap = info.get('marketCap', 0)
        if market_cap > 0:
            if market_cap >= 1e12:
                cap_display = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            else:
                cap_display = f"${market_cap/1e6:.1f}M"
            st.sidebar.write(f"**Market Cap:** {cap_display}")
        
        # Data quality indicator
        data_quality = calculate_data_quality_score(data, info)
        quality_emoji = "🟢" if data_quality >= 90 else "🟡" if data_quality >= 70 else "🔴"
        st.sidebar.write(f"**Data Quality:** {quality_emoji} {data_quality:.0f}%")

def calculate_data_quality_score(data, info):
    """Calculate data quality score based on available information"""
    score = 0
    max_score = 100
    
    # Basic data availability (40 points)
    if not data.empty: score += 20
    if len(data) > 30: score += 10  # Sufficient historical data
    if 'Volume' in data.columns and data['Volume'].sum() > 0: score += 10
    
    # Company info availability (30 points)
    key_info_fields = ['marketCap', 'trailingPE', 'sector', 'industry']
    available_fields = sum(1 for field in key_info_fields if info.get(field, 'N/A') != 'N/A')
    score += (available_fields / len(key_info_fields)) * 30
    
    # Financial data availability (30 points)
    if 'income_stmt' in st.session_state and st.session_state['income_stmt'] is not None:
        score += 15
    if 'balance_sheet' in st.session_state and st.session_state['balance_sheet'] is not None:
        score += 15
    
    return min(score, max_score)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_all_data(ticker, period, show_sma, show_bollinger):
    """Optimized data fetching with caching and error handling"""
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # Step 1: Get stock data
        status_text.text("📊 Fetching stock data...")
        progress_bar.progress(20)
        data, info = get_stock_data(ticker, period)
        
        if data is None or data.empty:
            st.error("❌ Failed to fetch stock data. Please check the ticker symbol.")
            return
        
        # Step 2: Calculate technical indicators
        status_text.text("📈 Calculating technical indicators...")
        progress_bar.progress(40)
        data = calculate_technical_indicators(data)
        
        # Step 3: Get financial statements
        status_text.text("💰 Fetching financial statements...")
        progress_bar.progress(60)
        income_stmt, balance_sheet, cash_flow = get_financial_statements(ticker)
        
        # Step 4: Get analyst data
        status_text.text("🎯 Fetching analyst recommendations...")
        progress_bar.progress(80)
        recommendations, price_targets = get_analyst_recommendations(ticker)
        
        # Step 5: Store everything in session state
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        st.session_state.update({
            'stock_data': data,
            'stock_info': info,
            'ticker': ticker,
            'show_sma': show_sma,
            'show_bollinger': show_bollinger,
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'recommendations': recommendations,
            'price_targets': price_targets,
            'last_updated': datetime.now()
        })
        
        st.sidebar.success(f"✅ Data loaded for {ticker}")
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def render_tabbed_dashboard():
    """Render main dashboard with organized tabs"""
    data = st.session_state['stock_data']
    info = st.session_state['stock_info']
    ticker = st.session_state['ticker']
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Technical Analysis", 
        "💰 Financial Performance", 
        "🎯 Analytics & Ratings", 
        "ℹ️ Company Info"
    ])
    
    with tab1:
        render_overview_tab(data, info, ticker)
    
    with tab2:
        render_technical_tab(data, ticker)
    
    with tab3:
        render_financial_tab()
    
    with tab4:
        render_analytics_tab(data, info, ticker)
    
    with tab5:
        render_company_tab(info)

def render_overview_tab(data, info, ticker):
    """Render overview tab with key metrics"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    current_price = data['Close'].iloc[-1]
    
    # Key metrics in a clean layout
    st.subheader(f"📈 {ticker} - Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        st.metric("Current Price", f"${current_price:.2f}", delta=f"{change_pct:+.2f}%")
    
    with col2:
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    with col3:
        pe_ratio = info.get('trailingPE', 'N/A')
        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio != 'N/A' else "N/A")
    
    with col4:
        dividend_yield = info.get('dividendYield', 0)
        dividend_yield_pct = dividend_yield * 100 if dividend_yield else 0
        st.metric("Dividend Yield", f"{dividend_yield_pct:.2f}%")
    
    # Price range and volume analysis
    render_price_range_slider(info, current_price)
    render_volume_analysis(data, info)
    render_short_interest_analysis(info)
    
    # Financial health dashboard
    if all(key in st.session_state for key in ['income_stmt', 'balance_sheet']):
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        ratios = calculate_financial_ratios(
            st.session_state['income_stmt'], 
            st.session_state['balance_sheet'], 
            info, 
            current_price, 
            shares_outstanding
        )
        render_financial_health_dashboard(ratios, info)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_technical_tab(data, ticker):
    """Enhanced render technical analysis tab with all available indicators"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("📊 Technical Analysis")
    
    # Enhanced technical indicators controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        show_sma = st.checkbox("Moving Averages", value=st.session_state.get('show_sma', False))
    with col2:
        show_bollinger = st.checkbox("Bollinger Bands", value=st.session_state.get('show_bollinger', False))
    with col3:
        show_volume = st.checkbox("Volume", value=st.session_state.get('show_volume', True))
    with col4:
        show_indicators = st.checkbox("Show RSI & MACD", value=st.session_state.get('show_indicators', False))
    
    # Main candlestick chart
    candlestick_fig = create_candlestick_chart(data, ticker, show_sma, show_bollinger, show_volume)
    st.plotly_chart(candlestick_fig, use_container_width=True)
    
    # Additional indicator charts if requested
    if show_indicators:
        col1, col2 = st.columns(2)
        with col1:
            if 'RSI' in data.columns:
                rsi_fig = create_rsi_chart(data, ticker)
                st.plotly_chart(rsi_fig, use_container_width=True)
        
        with col2:
            if 'MACD' in data.columns:
                macd_fig = create_macd_chart(data, ticker)
                st.plotly_chart(macd_fig, use_container_width=True)
    
    # Enhanced technical indicators summary
    render_technical_summary(data)
    
    # Trading signals section
    render_trading_signals(data)
    
    # Support/Resistance levels
    render_support_resistance(data)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_technical_summary(data):
    """Enhanced technical indicators summary with all available indicators"""
    st.subheader("📋 Technical Indicators Summary")
    
    # First row - Price action indicators
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    
    with col1:
        if 'SMA_20' in data.columns and pd.notna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_signal = "🟢 Above" if current_price > sma_20 else "🔴 Below"
            st.metric("vs SMA 20", sma_signal, f"${abs(current_price - sma_20):.2f}")
        else:
            st.metric("SMA 20", "N/A", "Calculating...")
    
    with col2:
        if 'SMA_50' in data.columns and pd.notna(data['SMA_50'].iloc[-1]):
            sma_50 = data['SMA_50'].iloc[-1]
            sma_signal = "🟢 Above" if current_price > sma_50 else "🔴 Below"
            st.metric("vs SMA 50", sma_signal, f"${abs(current_price - sma_50):.2f}")
        else:
            st.metric("SMA 50", "N/A", "Calculating...")
    
    with col3:
        if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                rsi_signal = "🔴 Overbought"
                rsi_color = "inverse"
            elif rsi < 30:
                rsi_signal = "🟢 Oversold"
                rsi_color = "normal"
            else:
                rsi_signal = "🟡 Neutral"
                rsi_color = "off"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal, delta_color=rsi_color)
        else:
            st.metric("RSI", "N/A", "Calculating...")
    
    with col4:
        # Volume trend
        if 'Volume_SMA' in data.columns and pd.notna(data['Volume_SMA'].iloc[-1]):
            volume_avg = data['Volume_SMA'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / volume_avg
            if volume_ratio > 1.5:
                volume_signal = "🟢 High"
                volume_color = "normal"
            elif volume_ratio < 0.5:
                volume_signal = "🔴 Low"
                volume_color = "inverse"
            else:
                volume_signal = "🟡 Normal"
                volume_color = "off"
            st.metric("Volume", volume_signal, f"{((volume_ratio - 1) * 100):+.1f}%", delta_color=volume_color)
        else:
            st.metric("Volume", "Normal", "0%")

    # Second row - Advanced indicators
    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # MACD Signal
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']) and pd.notna(data['MACD'].iloc[-1]):
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            if macd > macd_signal:
                macd_status = "🟢 Bullish"
                macd_color = "normal"
            else:
                macd_status = "🔴 Bearish"
                macd_color = "inverse"
            st.metric("MACD", macd_status, f"{(macd - macd_signal):.4f}", delta_color=macd_color)
        else:
            st.metric("MACD", "N/A", "Calculating...")
    
    with col6:
        # Bollinger Bands Position
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']) and pd.notna(data['BB_Middle'].iloc[-1]):
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_middle = data['BB_Middle'].iloc[-1]
            
            if current_price > bb_upper:
                bb_signal = "🔴 Above Upper"
                bb_color = "inverse"
            elif current_price < bb_lower:
                bb_signal = "🟢 Below Lower"
                bb_color = "normal"
            else:
                bb_signal = "🟡 In Range"
                bb_color = "off"
            
            # Calculate position within bands (0-100%)
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            st.metric("Bollinger Bands", bb_signal, f"{bb_position:.1f}%", delta_color=bb_color)
        else:
            st.metric("Bollinger Bands", "N/A", "Calculating...")
    
    with col7:
        # ATR (Volatility)
        if 'ATR' in data.columns and pd.notna(data['ATR'].iloc[-1]):
            atr = data['ATR'].iloc[-1]
            atr_pct = (atr / current_price) * 100
            if atr_pct > 3:
                atr_signal = "🔴 High Volatility"
            elif atr_pct < 1:
                atr_signal = "🟢 Low Volatility"
            else:
                atr_signal = "🟡 Normal Volatility"
            st.metric("ATR", atr_signal, f"{atr_pct:.2f}%")
        else:
            st.metric("ATR", "N/A", "Calculating...")
    
    with col8:
        # Price Change
        if 'Price_Change' in data.columns and pd.notna(data['Price_Change'].iloc[-1]):
            price_change = data['Price_Change'].iloc[-1] * 100
            if abs(price_change) > 2:
                change_signal = "🔴 High Move" if price_change > 0 else "🔴 Sharp Drop"
                change_color = "normal" if price_change > 0 else "inverse"
            else:
                change_signal = "🟡 Normal"
                change_color = "off"
            st.metric("Daily Change", change_signal, f"{price_change:+.2f}%", delta_color=change_color)
        else:
            daily_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            st.metric("Daily Change", "🟡 Normal", f"{daily_change:+.2f}%")

def render_trading_signals(data):
    """Render trading signals section"""
    st.subheader("🎯 Trading Signals")
    
    signals = get_technical_signals(data)
    
    if signals:
        # Create columns for different signal types
        signal_cols = st.columns(len(signals))
        
        for i, (indicator, signal) in enumerate(signals.items()):
            with signal_cols[i]:
                # Color coding for signals
                if 'Bullish' in signal or 'Buy' in signal or 'Oversold' in signal:
                    st.success(f"**{indicator}**\n\n{signal}")
                elif 'Bearish' in signal or 'Sell' in signal or 'Overbought' in signal:
                    st.error(f"**{indicator}**\n\n{signal}")
                else:
                    st.info(f"**{indicator}**\n\n{signal}")
    else:
        st.info("Calculating trading signals...")

def render_support_resistance(data):
    """Render support and resistance levels"""
    st.subheader("📊 Support & Resistance Levels")
    
    try:
        levels = get_support_resistance_levels(data)
        current_price = data['Close'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            resistance_diff = levels['strong_resistance'] - current_price
            st.metric(
                "Strong Resistance", 
                f"${levels['strong_resistance']:.2f}",
                f"+${resistance_diff:.2f} ({(resistance_diff/current_price)*100:+.1f}%)"
            )
        
        with col2:
            resistance_diff = levels['resistance'] - current_price
            st.metric(
                "Resistance", 
                f"${levels['resistance']:.2f}",
                f"+${resistance_diff:.2f} ({(resistance_diff/current_price)*100:+.1f}%)"
            )
        
        with col3:
            support_diff = current_price - levels['support']
            st.metric(
                "Support", 
                f"${levels['support']:.2f}",
                f"-${support_diff:.2f} ({(support_diff/current_price)*100:+.1f}%)"
            )
        
        with col4:
            strong_support_diff = current_price - levels['strong_support']
            st.metric(
                "Strong Support", 
                f"${levels['strong_support']:.2f}",
                f"-${strong_support_diff:.2f} ({(strong_support_diff/current_price)*100:+.1f}%)"
            )
            
    except Exception as e:
        st.info("Calculating support and resistance levels...")

def render_volatility_metrics(data):
    """Render volatility metrics (optional addition)"""
    st.subheader("📈 Volatility Analysis")
    
    try:
        volatility_metrics = calculate_volatility_metrics(data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_vol = volatility_metrics['daily_volatility'] * 100
            st.metric("Daily Volatility", f"{daily_vol:.2f}%")
        
        with col2:
            annual_vol = volatility_metrics['annualized_volatility'] * 100
            st.metric("Annualized Volatility", f"{annual_vol:.1f}%")
        
        with col3:
            if volatility_metrics['average_true_range']:
                atr_val = volatility_metrics['average_true_range']
                st.metric("Average True Range", f"${atr_val:.2f}")
            else:
                st.metric("Average True Range", "Calculating...")
                
    except Exception as e:
        st.info("Calculating volatility metrics...")

def render_financial_tab():
    """Render financial performance tab"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if 'income_stmt' not in st.session_state or st.session_state['income_stmt'] is None:
        st.info("📊 Financial statement data not available for this ticker")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.subheader("💰 Financial Performance")
    
    # Create sub-tabs for better organization
    fin_tab1, fin_tab2, fin_tab3 = st.tabs(["📊 Statements", "📈 Historical Trends", "🔍 Advanced Analysis"])
    
    with fin_tab1:
        render_financial_statements_subtab()
    
    with fin_tab2:
        render_historical_trends_subtab()
    
    with fin_tab3:
        render_advanced_analysis_subtab()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_financial_statements_subtab():
    """Render financial statements sub-tab"""
    income_stmt = st.session_state.get('income_stmt')
    balance_sheet = st.session_state.get('balance_sheet')
    
    if income_stmt is not None:
        combined_income_fig = create_combined_income_chart(income_stmt)
        if combined_income_fig:
            st.plotly_chart(combined_income_fig, use_container_width=True)
    
    if balance_sheet is not None:
        combined_balance_fig = create_combined_balance_sheet_chart(balance_sheet)
        if combined_balance_fig:
            st.plotly_chart(combined_balance_fig, use_container_width=True)

def render_historical_trends_subtab():
    """Render historical trends sub-tab"""
    income_stmt = st.session_state.get('income_stmt')
    balance_sheet = st.session_state.get('balance_sheet')
    
    if income_stmt is not None and balance_sheet is not None:
        historical_ratios_fig = create_historical_ratios_chart(income_stmt, balance_sheet)
        if historical_ratios_fig:
            st.plotly_chart(historical_ratios_fig, use_container_width=True)

def render_advanced_analysis_subtab():
    """Render advanced analysis sub-tab"""
    income_stmt = st.session_state.get('income_stmt')
    
    if income_stmt is None or income_stmt.empty:
        st.info("Income statement data required for advanced analysis")
        return
    
    available_years = [str(col)[:4] for col in income_stmt.columns]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💧 Waterfall Analysis")
        selected_year_waterfall = st.selectbox("Select Year", available_years, key="waterfall_year")
        waterfall_fig = create_waterfall_chart(income_stmt, selected_year_waterfall)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
    
    with col2:
        st.subheader("🌊 Flow Analysis")
        selected_year_sankey = st.selectbox("Select Year", available_years, key="sankey_year")
        sankey_fig = create_sankey_diagram(income_stmt, selected_year_sankey)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)

def render_analytics_tab(data, info, ticker):
    """Render analytics and ratings tab"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    current_price = data['Close'].iloc[-1]
    
    # Create sub-tabs for analytics
    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "📈 EPS Analysis", 
        "💰 Dividend Analysis", 
        "🎯 Analyst Ratings", 
        "🏭 Industry Comparison"
    ])
    
    with analytics_tab1:
        render_eps_analysis_subtab(info)
    
    with analytics_tab2:
        render_dividend_analysis_subtab(ticker, info)
    
    with analytics_tab3:
        render_analyst_ratings_subtab(current_price)
    
    with analytics_tab4:
        render_industry_comparison_subtab(ticker, info, current_price)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_eps_analysis_subtab(info):
    """Render EPS analysis sub-tab"""
    if 'income_stmt' not in st.session_state or st.session_state['income_stmt'] is None:
        st.info("📊 Income statement data required for EPS analysis")
        return
    
    st.subheader("📈 Earnings Per Share Analysis")
    
    eps_fig = create_eps_trend_chart(st.session_state['income_stmt'], info)
    
    if eps_fig:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(eps_fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 EPS Metrics**")
            
            trailing_eps = info.get('trailingEps', 'N/A')
            forward_eps = info.get('forwardEps', 'N/A')
            peg_ratio = info.get('pegRatio', 'N/A')
            
            if trailing_eps != 'N/A':
                st.metric("Trailing EPS", f"${trailing_eps:.2f}")
            
            if forward_eps != 'N/A':
                st.metric("Forward EPS", f"${forward_eps:.2f}")
                
                # EPS growth calculation
                if trailing_eps != 'N/A' and trailing_eps > 0:
                    eps_growth = ((forward_eps - trailing_eps) / trailing_eps * 100)
                    growth_color = "🟢" if eps_growth > 0 else "🔴"
                    st.metric("EPS Growth", f"{growth_color} {eps_growth:+.1f}%")
            
            if peg_ratio != 'N/A':
                peg_interpretation = "🟢 Undervalued" if peg_ratio < 1 else "🟡 Fair" if peg_ratio < 2 else "🔴 Overvalued"
                st.metric("PEG Ratio", f"{peg_ratio:.2f}")
                st.caption(peg_interpretation)
    else:
        st.info("EPS trend data not available")

def render_dividend_analysis_subtab(ticker, info):
    """Render dividend analysis sub-tab"""
    st.subheader("💰 Dividend Analysis")
    
    dividend_fig = create_dividend_history_chart(ticker)
    
    if dividend_fig:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(dividend_fig, use_container_width=True)
        
        with col2:
            render_dividend_metrics_detailed(info)
    else:
        render_basic_dividend_info_enhanced(info)

def render_dividend_metrics_detailed(info):
    """Render detailed dividend metrics"""
    st.markdown("**💰 Dividend Metrics**")
    
    dividend_rate = info.get('dividendRate', 'N/A')
    dividend_yield = info.get('dividendYield', 0)
    payout_ratio = info.get('payoutRatio', 'N/A')
    ex_dividend_date = info.get('exDividendDate', 'N/A')
    
    # Annual dividend
    if dividend_rate != 'N/A':
        st.metric("Annual Dividend", f"${dividend_rate:.2f}")
    
    # Dividend yield with interpretation
    yield_pct = dividend_yield * 100 if dividend_yield else 0
    if yield_pct > 4:
        yield_status = "🟢 High Yield"
    elif yield_pct > 2:
        yield_status = "🟡 Moderate"
    elif yield_pct > 0:
        yield_status = "🔴 Low Yield"
    else:
        yield_status = "❌ No Dividend"
    
    st.metric("Dividend Yield", f"{yield_pct:.2f}%")
    st.caption(yield_status)
    
    # Payout ratio with sustainability indicator
    if payout_ratio != 'N/A':
        payout_pct = payout_ratio * 100
        if payout_pct < 50:
            sustainability = "🟢 Very Sustainable"
        elif payout_pct < 70:
            sustainability = "🟡 Sustainable"
        elif payout_pct < 90:
            sustainability = "🟠 Moderate Risk"
        else:
            sustainability = "🔴 High Risk"
        
        st.metric("Payout Ratio", f"{payout_pct:.1f}%")
        st.caption(f"Sustainability: {sustainability}")
    
    # Ex-dividend date
    if ex_dividend_date != 'N/A':
        st.write(f"**Ex-Dividend Date:** {ex_dividend_date}")

def render_basic_dividend_info_enhanced(info):
    """Render enhanced basic dividend information"""
    dividend_yield = info.get('dividendYield', 0)
    dividend_rate = info.get('dividendRate', 'N/A')
    
    if dividend_rate != 'N/A' or dividend_yield > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if dividend_rate != 'N/A':
                st.metric("Annual Dividend", f"${dividend_rate:.2f}")
            else:
                st.metric("Annual Dividend", "N/A")
        
        with col2:
            yield_pct = dividend_yield * 100 if dividend_yield else 0
            st.metric("Dividend Yield", f"{yield_pct:.2f}%")
        
        with col3:
            payout_ratio = info.get('payoutRatio', 'N/A')
            if payout_ratio != 'N/A':
                st.metric("Payout Ratio", f"{payout_ratio*100:.1f}%")
            else:
                st.metric("Payout Ratio", "N/A")
    else:
        st.info("💡 This stock does not pay dividends or dividend data is not available")
        
        # Show alternative return metrics
        col1, col2 = st.columns(2)
        with col1:
            # Calculate price appreciation
            if 'stock_data' in st.session_state:
                data = st.session_state['stock_data']
                if len(data) > 252:  # 1 year of data
                    year_ago_price = data['Close'].iloc[-252]
                    current_price = data['Close'].iloc[-1]
                    price_return = ((current_price - year_ago_price) / year_ago_price * 100)
                    st.metric("1-Year Price Return", f"{price_return:+.1f}%")
        
        with col2:
            # Show total return potential
            st.write("**Focus on:**")
            st.write("• Capital appreciation")
            st.write("• Share buybacks")
            st.write("• Growth reinvestment")

def render_analyst_ratings_subtab(current_price):
    """Render analyst ratings sub-tab"""
    st.subheader("🎯 Analyst Recommendations")
    
    recommendations = st.session_state.get('recommendations')
    price_targets = st.session_state.get('price_targets')
    
    # Check if recommendations data exists and is valid
    has_recommendations = (
        recommendations is not None and 
        (
            (hasattr(recommendations, 'empty') and not recommendations.empty) or
            (isinstance(recommendations, dict) and recommendations) or
            (isinstance(recommendations, list) and len(recommendations) > 0)
        )
    )
    
    if has_recommendations:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                analyst_fig, rec_counts = display_analyst_ratings(recommendations, price_targets, current_price)
                if analyst_fig:
                    st.plotly_chart(analyst_fig, use_container_width=True)
                else:
                    st.info("📊 Unable to generate analyst ratings chart")
            except Exception as e:
                st.error(f"Error generating analyst chart: {str(e)}")
                analyst_fig, rec_counts = None, None
        
        with col2:
            try:
                render_analyst_summary_enhanced(rec_counts, price_targets, current_price)
            except Exception as e:
                st.error(f"Error displaying analyst summary: {str(e)}")
                st.info("📊 Analyst summary data not available")
    else:
        st.info("📊 Analyst recommendation data not available for this ticker")
        
        # Show alternative analysis
        st.markdown("**💡 Alternative Analysis:**")
        st.write("• Check financial health score")
        st.write("• Review technical indicators")
        st.write("• Compare industry metrics")

def render_analyst_summary_enhanced(rec_counts, price_targets, current_price):
    """Render enhanced analyst summary"""
    st.markdown("**📊 Rating Distribution**")
    
    if rec_counts is not None:
        total_ratings = rec_counts.sum()
        
        # Create a more visual representation
        for rating, count in rec_counts.items():
            percentage = (count / total_ratings * 100) if total_ratings > 0 else 0
            
            # Color coding for ratings
            if rating.lower() in ['strong buy', 'buy']:
                emoji = "🟢"
            elif rating.lower() in ['hold', 'neutral']:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            st.write(f"{emoji} **{rating}**: {count} ({percentage:.1f}%)")
    
    # Price targets analysis - FIXED: Check if it's a dict and has data
    if price_targets is not None and isinstance(price_targets, dict) and price_targets:
        st.markdown("---")
        st.markdown("**🎯 Price Target Analysis**")
        
        target_mean = price_targets.get('targetMeanPrice', 'N/A')
        target_high = price_targets.get('targetHighPrice', 'N/A')
        target_low = price_targets.get('targetLowPrice', 'N/A')
        
        if target_mean != 'N/A' and target_mean is not None:
            upside = ((target_mean - current_price) / current_price * 100)
            upside_color = "🟢" if upside > 0 else "🔴"
            st.metric("Avg Target", f"${target_mean:.2f}", 
                     delta=f"{upside_color} {upside:+.1f}%")
        
        # Target range
        if (target_high != 'N/A' and target_high is not None and 
            target_low != 'N/A' and target_low is not None):
            st.write(f"**Range:** ${target_low:.2f} - ${target_high:.2f}")
            
            # Position within target range
            if target_high > target_low:
                position = (current_price - target_low) / (target_high - target_low)
                position_pct = position * 100
                
                if position_pct < 25:
                    position_status = "🔴 Near Low Target"
                elif position_pct > 75:
                    position_status = "🟢 Near High Target"
                else:
                    position_status = "🟡 Mid-Range"
                
                st.caption(f"Position: {position_status} ({position_pct:.1f}%)")
    elif price_targets is not None and hasattr(price_targets, 'empty') and not price_targets.empty:
        # Handle case where price_targets is a DataFrame
        st.markdown("---")
        st.markdown("**🎯 Price Target Analysis**")
        
        # If it's a DataFrame, get the first row or appropriate data
        if len(price_targets) > 0:
            target_data = price_targets.iloc[0] if hasattr(price_targets, 'iloc') else price_targets
            
            target_mean = target_data.get('targetMeanPrice', 'N/A')
            target_high = target_data.get('targetHighPrice', 'N/A')
            target_low = target_data.get('targetLowPrice', 'N/A')
            
            if target_mean != 'N/A' and target_mean is not None:
                upside = ((target_mean - current_price) / current_price * 100)
                upside_color = "🟢" if upside > 0 else "🔴"
                st.metric("Avg Target", f"${target_mean:.2f}", 
                         delta=f"{upside_color} {upside:+.1f}%")
            
            # Target range
            if (target_high != 'N/A' and target_high is not None and 
                target_low != 'N/A' and target_low is not None):
                st.write(f"**Range:** ${target_low:.2f} - ${target_high:.2f}")
    else:
        st.info("📊 Price target data not available")

def render_industry_comparison_subtab(ticker, info, current_price):
    """Render industry comparison sub-tab"""
    st.subheader("🏭 Industry Comparison")
    
    industry_benchmarks = get_industry_comparison(ticker, info)
    sector = info.get('sector', 'Unknown')
    industry = info.get('industry', 'Unknown')
    
    # Get current metrics
    current_metrics = {
        'pe_ratio': info.get('trailingPE', 0),
        'pb_ratio': info.get('priceToBook', 0),
        'roe': 0
    }
    
    # Calculate ROE if financial data is available
    if all(key in st.session_state for key in ['income_stmt', 'balance_sheet']):
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        ratios = calculate_financial_ratios(
            st.session_state['income_stmt'], 
            st.session_state['balance_sheet'], 
            info, 
            current_price, 
            shares_outstanding
        )
        current_metrics['roe'] = ratios.get('ROE %', 0)
    
    # Industry comparison chart
    industry_fig = create_industry_comparison_chart(current_metrics, industry_benchmarks, sector)
    
    if industry_fig:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(industry_fig, use_container_width=True)
        
        with col2:
            render_industry_benchmarks_enhanced(industry_benchmarks, current_metrics, sector, industry)
    else:
        st.info("📊 Industry comparison data not available")

def render_industry_benchmarks_enhanced(industry_benchmarks, current_metrics, sector, industry):
    """Render enhanced industry benchmark information"""
    st.markdown(f"**📋 {sector} Sector**")
    st.caption(f"Industry: {industry}")
    
    # Benchmark metrics
    st.markdown("**Industry Averages:**")
    st.write(f"• P/E Ratio: {industry_benchmarks['avg_pe']:.1f}")
    st.write(f"• P/B Ratio: {industry_benchmarks['avg_pb']:.1f}")
    st.write(f"• ROE: {industry_benchmarks['avg_roe']:.1f}%")
    
    st.markdown("---")
    st.markdown("**📈 Relative Performance**")
    
    # P/E comparison
    pe_ratio = current_metrics['pe_ratio']
    if pe_ratio and pe_ratio > 0:
        pe_diff = pe_ratio - industry_benchmarks['avg_pe']
        pe_pct_diff = (pe_diff / industry_benchmarks['avg_pe'] * 100) if industry_benchmarks['avg_pe'] > 0 else 0
        
        if pe_pct_diff > 20:
            pe_status = "🔴 Much Higher"
        elif pe_pct_diff > 0:
            pe_status = "🟡 Above Average"
        elif pe_pct_diff > -20:
            pe_status = "🟢 Below Average"
        else:
            pe_status = "🟢 Much Lower"
        
        st.write(f"P/E: {pe_status} ({pe_pct_diff:+.1f}%)")
    
    # ROE comparison
    roe = current_metrics['roe']
    if roe and roe != 0:
        roe_diff = roe - industry_benchmarks['avg_roe']
        
        if roe_diff > 5:
            roe_status = "🟢 Much Better"
        elif roe_diff > 0:
            roe_status = "🟢 Above Average"
        elif roe_diff > -5:
            roe_status = "🟡 Below Average"
        else:
            roe_status = "🔴 Much Worse"
        
        st.write(f"ROE: {roe_status} ({roe_diff:+.1f}pp)")
    
    # Overall assessment
    st.markdown("---")
    
    # Simple scoring system
    score = 0
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < industry_benchmarks['avg_pe']:
            score += 1
    if roe and roe > industry_benchmarks['avg_roe']:
        score += 1
    
    if score >= 2:
        overall_status = "🟢 Outperforming Industry"
    elif score == 1:
        overall_status = "🟡 Mixed Performance"
    else:
        overall_status = "🔴 Underperforming Industry"
    
    st.markdown(f"**Overall:** {overall_status}")

def render_company_tab(info):
    """Render company information tab"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if not info:
        st.info("📊 Company information not available")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.subheader("ℹ️ Company Information")
    
    # Basic company info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏢 Company Details**")
        st.write(f"**Name:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
        st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
        
        website = info.get('website', 'N/A')
        if website != 'N/A':
            st.markdown(f"**Website:** [{website}]({website})")
    
    with col2:
        st.markdown("**👥 Company Stats**")
        
        employees = info.get('fullTimeEmployees', 'N/A')
        if employees != 'N/A':
            st.write(f"**Employees:** {employees:,}")
        else:
            st.write("**Employees:** N/A")
        
        # Market metrics
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            if market_cap >= 1e12:
                cap_category = "Mega Cap (>$1T)"
            elif market_cap >= 200e9:
                cap_category = "Large Cap ($200B-$1T)"
            elif market_cap >= 10e9:
                cap_category = "Large Cap ($10B-$200B)"
            elif market_cap >= 2e9:
                cap_category = "Mid Cap ($2B-$10B)"
            elif market_cap >= 300e6:
                cap_category = "Small Cap ($300M-$2B)"
            else:
                cap_category = "Micro Cap (<$300M)"
            
            st.write(f"**Market Cap Category:** {cap_category}")
        
        # Founded year (if available)
        founded = info.get('founded', info.get('firstTradeDateEpochUtc', 'N/A'))
        if founded != 'N/A':
            if isinstance(founded, (int, float)):
                if founded > 1e9:  # Epoch timestamp
                    founded_year = datetime.fromtimestamp(founded).year
                else:
                    founded_year = int(founded)
                st.write(f"**Founded/Listed:** {founded_year}")
        
        # ESG Score (if available)
        esg_score = info.get('esgScores', {}).get('totalEsg', 'N/A') if info.get('esgScores') else 'N/A'
        if esg_score != 'N/A':
            st.write(f"**ESG Score:** {esg_score:.1f}")
    
    # Business summary
    if 'longBusinessSummary' in info and info['longBusinessSummary']:
        st.markdown("---")
        st.subheader("📋 Business Summary")
        
        # Make summary collapsible for better UX
        with st.expander("View Business Summary", expanded=False):
            st.write(info['longBusinessSummary'])
    
    # Key executives (if available)
    if 'companyOfficers' in info and info['companyOfficers']:
        st.markdown("---")
        st.subheader("👔 Key Executives")
        
        officers = info['companyOfficers'][:5]  # Show top 5 executives
        
        for officer in officers:
            name = officer.get('name', 'N/A')
            title = officer.get('title', 'N/A')
            age = officer.get('age', 'N/A')
            
            col_exec1, col_exec2 = st.columns([2, 1])
            with col_exec1:
                st.write(f"**{name}**")
                st.caption(title)
            with col_exec2:
                if age != 'N/A':
                    st.caption(f"Age: {age}")
    
    # Financial highlights
    render_company_financial_highlights(info)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_company_financial_highlights(info):
    """Render company financial highlights"""
    st.markdown("---")
    st.subheader("💼 Financial Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Valuation Metrics**")
        
        # P/E ratios
        trailing_pe = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        
        if trailing_pe != 'N/A':
            st.write(f"Trailing P/E: {trailing_pe:.2f}")
        if forward_pe != 'N/A':
            st.write(f"Forward P/E: {forward_pe:.2f}")
        
        # Book value
        book_value = info.get('bookValue', 'N/A')
        if book_value != 'N/A':
            st.write(f"Book Value: ${book_value:.2f}")
        
        # Price to book
        price_to_book = info.get('priceToBook', 'N/A')
        if price_to_book != 'N/A':
            st.write(f"P/B Ratio: {price_to_book:.2f}")
    
    with col2:
        st.markdown("**💰 Profitability**")
        
        # Margins
        profit_margins = info.get('profitMargins', 'N/A')
        if profit_margins != 'N/A':
            st.write(f"Profit Margin: {profit_margins*100:.1f}%")
        
        gross_margins = info.get('grossMargins', 'N/A')
        if gross_margins != 'N/A':
            st.write(f"Gross Margin: {gross_margins*100:.1f}%")
        
        operating_margins = info.get('operatingMargins', 'N/A')
        if operating_margins != 'N/A':
            st.write(f"Operating Margin: {operating_margins*100:.1f}%")
        
        # Return metrics
        return_on_assets = info.get('returnOnAssets', 'N/A')
        if return_on_assets != 'N/A':
            st.write(f"ROA: {return_on_assets*100:.1f}%")
        
        return_on_equity = info.get('returnOnEquity', 'N/A')
        if return_on_equity != 'N/A':
            st.write(f"ROE: {return_on_equity*100:.1f}%")
    
    with col3:
        st.markdown("**📈 Growth & Efficiency**")
        
        # Growth rates
        earnings_growth = info.get('earningsGrowth', 'N/A')
        if earnings_growth != 'N/A':
            st.write(f"Earnings Growth: {earnings_growth*100:.1f}%")
        
        revenue_growth = info.get('revenueGrowth', 'N/A')
        if revenue_growth != 'N/A':
            st.write(f"Revenue Growth: {revenue_growth*100:.1f}%")
        
        # Efficiency metrics
        asset_turnover = info.get('assetTurnover', 'N/A')
        if asset_turnover != 'N/A':
            st.write(f"Asset Turnover: {asset_turnover:.2f}")
        
        # Beta
        beta = info.get('beta', 'N/A')
        if beta != 'N/A':
            risk_level = "Low" if beta < 0.8 else "Moderate" if beta < 1.2 else "High"
            st.write(f"Beta: {beta:.2f} ({risk_level} Risk)")

def render_welcome_screen():
    """Enhanced welcome screen with better guidance"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🚀 Welcome to the Stock Financial Dashboard</h2>
        <p style="font-size: 1.2em; color: #666;">
            Get comprehensive financial analysis for any publicly traded stock
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start guide
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Quick Start")
        st.markdown("""
        1. **Enter a ticker** in the sidebar (e.g., AAPL, GOOGL)
        2. **Select time period** for historical data
        3. **Choose technical indicators** to display
        4. **Click 'Fetch Data'** to load the dashboard
        """)
        
        st.markdown("### 📊 What You'll Get")
        st.markdown("""
        - **Real-time stock data** and key metrics
        - **Technical analysis** with interactive charts
        - **Financial statements** and ratio analysis
        - **Analyst ratings** and price targets
        - **Industry comparisons** and benchmarks
        """)
    
    with col2:
        st.markdown("### 🔥 Popular Tickers")
        
        # Organized by category with better presentation
        categories = {
            "💻 Technology": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META"],
            "🏦 Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
            "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "BMY"],
            "🛒 Consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "COST"],
            "🏭 Industrial": ["BA", "CAT", "GE", "MMM", "HON", "UPS"],
            "⚡ Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC"]
        }
        
        for category, tickers in categories.items():
            with st.expander(category, expanded=False):
                # Create clickable ticker buttons
                cols = st.columns(3)
                for i, ticker in enumerate(tickers):
                    with cols[i % 3]:
                        if st.button(ticker, key=f"example_{ticker}"):
                            # Auto-fill the sidebar with selected ticker
                            st.session_state['auto_ticker'] = ticker
                            st.rerun()
    
    # Features highlight
    st.markdown("---")
    st.markdown("### ✨ Dashboard Features")
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown("""
        **📈 Overview**
        - Current price & metrics
        - 52-week range analysis
        - Volume & liquidity
        - Financial health score
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **📊 Technical Analysis**
        - Interactive candlestick charts
        - Moving averages
        - Bollinger bands
        - Technical indicators
        """)
    
    with feature_cols[2]:
        st.markdown("""
        **💰 Financial Performance**
        - Income statements
        - Balance sheets
        - Historical trends
        - Advanced visualizations
        """)
    
    with feature_cols[3]:
        st.markdown("""
        **🎯 Analytics & Ratings**
        - EPS analysis
        - Dividend history
        - Analyst recommendations
        - Industry comparisons
        """)
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.info("""
        **📊 For Best Analysis:**
        - Use 1-year+ periods for trend analysis
        - Compare with industry benchmarks
        - Check multiple timeframes
        - Review analyst consensus
        """)
    
    with tips_col2:
        st.warning("""
        **⚠️ Important Notes:**
        - Data is for informational purposes only
        - Not financial advice
        - Always do your own research
        - Consider multiple data sources
        """)
# Auto-fill ticker if selected from examples
if 'auto_ticker' in st.session_state:
    st.session_state['example_ticker'] = st.session_state['auto_ticker']
    del st.session_state['auto_ticker']

if __name__ == "__main__":
    main()