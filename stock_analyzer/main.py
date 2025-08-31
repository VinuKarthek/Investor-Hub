"""
Stock Financial Dashboard - Main Application
Run this file to start the dashboard: streamlit run main.py
"""

import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.data_fetcher import get_stock_data, get_financial_statements, get_analyst_recommendations
from modules.technical_analysis import calculate_technical_indicators, create_candlestick_chart
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
    render_financial_ratios
)

# Page configuration
st.set_page_config(
    page_title="Stock Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e6e6e6;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    st.markdown('<h1 class="main-header">📊 Stock Financial Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar Controls
    render_sidebar()
    
    # Main Dashboard Content
    if 'stock_data' in st.session_state:
        render_dashboard()
    else:
        render_welcome_screen()

def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.header("Dashboard Controls")
    
    # Stock ticker input
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker", 
        value="AAPL", 
        help="e.g., AAPL, GOOGL, MSFT, TSLA"
    )
    
    # Time period selection
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.sidebar.selectbox("Select Time Period", period_options, index=3)
    
    # Technical indicators
    st.sidebar.subheader("Technical Indicators")
    show_sma = st.sidebar.checkbox("Show Moving Averages (SMA 20, 50)")
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands")
    
    # Fetch data button
    if st.sidebar.button("Fetch Data", type="primary"):
        if ticker:
            fetch_all_data(ticker.upper(), period, show_sma, show_bollinger)

def fetch_all_data(ticker, period, show_sma, show_bollinger):
    """Fetch all required data for the dashboard"""
    with st.spinner("Fetching comprehensive stock data..."):
        # Get stock data
        data, info = get_stock_data(ticker, period)
        
        if data is not None and not data.empty:
            # Calculate technical indicators
            data = calculate_technical_indicators(data)
            
            # Store in session state
            st.session_state.update({
                'stock_data': data,
                'stock_info': info,
                'ticker': ticker,
                'show_sma': show_sma,
                'show_bollinger': show_bollinger
            })
            
            # Get financial statements
            income_stmt, balance_sheet, cash_flow = get_financial_statements(ticker)
            st.session_state.update({
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            })
            
            # Get analyst data
            recommendations, price_targets = get_analyst_recommendations(ticker)
            st.session_state.update({
                'recommendations': recommendations,
                'price_targets': price_targets
            })
            
            st.success(f"✅ All data fetched successfully for {ticker}!")
        else:
            st.error("❌ Failed to fetch stock data. Please check the ticker symbol.")

def render_dashboard():
    """Render the main dashboard content"""
    data = st.session_state['stock_data']
    info = st.session_state['stock_info']
    ticker = st.session_state['ticker']
    
    # Basic Statistics Section
    render_basic_statistics(data, info, ticker)
    
    # Advanced Analytics Section
    render_advanced_analytics(data, info, ticker)
    
    # Technical Analysis Section
    render_technical_analysis(data, ticker)
    
    # Financial Performance Section
    render_financial_performance()
    
    # Company Information Section
    render_company_information(info)

def render_basic_statistics(data, info, ticker):
    """Render basic statistics section"""
    st.header(f"📈 {ticker} - Basic Statistics")
    
    current_price = data['Close'].iloc[-1]
    
    # Core metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
        )
    
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
    
    # 52-Week Range Slider
    render_price_range_slider(info, current_price)
    
    # Volume Analysis
    render_volume_analysis(data, info)
    
    # Short Interest Analysis
    render_short_interest_analysis(info)
    
    # Financial Ratios
    if 'income_stmt' in st.session_state and 'balance_sheet' in st.session_state:
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        ratios = calculate_financial_ratios(
            st.session_state['income_stmt'], 
            st.session_state['balance_sheet'], 
            info, 
            current_price, 
            shares_outstanding
        )
        render_financial_ratios(ratios)

def render_advanced_analytics(data, info, ticker):
    """Render advanced analytics section"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    current_price = data['Close'].iloc[-1]
    
    # EPS Analysis
    if 'income_stmt' in st.session_state:
        st.subheader("📈 Earnings Per Share (EPS) Analysis")
        eps_fig = create_eps_trend_chart(st.session_state['income_stmt'], info)
        if eps_fig:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(eps_fig, use_container_width=True)
            with col2:
                render_eps_metrics(info, st.session_state['income_stmt'])
    
    # Dividend Analysis
    st.subheader("💰 Dividend Analysis")
    render_dividend_analysis(ticker, info)
    
    # Analyst Ratings
    if 'recommendations' in st.session_state:
        st.subheader("🎯 Analyst Recommendations")
        render_analyst_section(current_price)
    
    # Industry Comparison
    st.subheader("🏭 Industry Comparison")
    render_industry_comparison(ticker, info, current_price)

def render_eps_metrics(info, income_stmt):
    """Render EPS metrics"""
    trailing_eps = info.get('trailingEps', 'N/A')
    forward_eps = info.get('forwardEps', 'N/A')
    peg_ratio = info.get('pegRatio', 'N/A')
    
    st.metric("Trailing EPS", f"${trailing_eps:.2f}" if trailing_eps != 'N/A' else "N/A")
    st.metric("Forward EPS", f"${forward_eps:.2f}" if forward_eps != 'N/A' else "N/A")
    st.metric("PEG Ratio", f"{peg_ratio:.2f}" if peg_ratio != 'N/A' else "N/A")

def render_dividend_analysis(ticker, info):
    """Render dividend analysis section"""
    dividend_fig = create_dividend_history_chart(ticker)
    if dividend_fig:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(dividend_fig, use_container_width=True)
        with col2:
            render_dividend_metrics(info)
    else:
        render_basic_dividend_info(info)

def render_dividend_metrics(info):
    """Render dividend metrics"""
    dividend_rate = info.get('dividendRate', 'N/A')
    dividend_yield = info.get('dividendYield', 0)
    payout_ratio = info.get('payoutRatio', 'N/A')
    
    st.metric("Annual Dividend", f"${dividend_rate:.2f}" if dividend_rate != 'N/A' else "N/A")
    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "0.00%")
    st.metric("Payout Ratio", f"{payout_ratio*100:.1f}%" if payout_ratio != 'N/A' else "N/A")
    
    # Sustainability indicator
    if payout_ratio != 'N/A':
        if payout_ratio < 0.5:
            sustainability = "🟢 Sustainable"
        elif payout_ratio < 0.8:
            sustainability = "🟡 Moderate Risk"
        else:
            sustainability = "🔴 High Risk"
        st.caption(f"Sustainability: {sustainability}")

def render_basic_dividend_info(info):
    """Render basic dividend information when no history is available"""
    dividend_yield = info.get('dividendYield', 0)
    dividend_rate = info.get('dividendRate', 'N/A')
    
    if dividend_rate != 'N/A' or dividend_yield > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Dividend", f"${dividend_rate:.2f}" if dividend_rate != 'N/A' else "N/A")
        with col2:
            st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "0.00%")
        with col3:
            payout_ratio = info.get('payoutRatio', 'N/A')
            st.metric("Payout Ratio", f"{payout_ratio*100:.1f}%" if payout_ratio != 'N/A' else "N/A")
    else:
        st.info("This stock does not pay dividends or dividend data is not available")

def render_analyst_section(current_price):
    """Render analyst recommendations section"""
    recommendations = st.session_state.get('recommendations')
    price_targets = st.session_state.get('price_targets')
    
    if recommendations is not None and not recommendations.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            analyst_fig, rec_counts = display_analyst_ratings(recommendations, price_targets, current_price)
            if analyst_fig:
                st.plotly_chart(analyst_fig, use_container_width=True)
        #TODO
        #with col2:
        #    render_analyst_summary(rec_counts, price_targets, current_price)
    else:
        st.info("Analyst recommendation data not available")

def render_analyst_summary(rec_counts, price_targets, current_price):
    """Render analyst summary"""
    st.markdown("**📊 Rating Summary**")
    if rec_counts is not None:
        total_ratings = rec_counts.sum()
        for rating, count in rec_counts.items():
            percentage = (count / total_ratings * 100)
            st.write(f"**{rating}**: {count} ({percentage:.1f}%)")
    
    # Price targets
    if price_targets is not None and not price_targets.empty:
        st.markdown("**🎯 Price Targets**")
        target_mean = price_targets.get('targetMeanPrice', 'N/A')
        target_high = price_targets.get('targetHighPrice', 'N/A')
        target_low = price_targets.get('targetLowPrice', 'N/A')
        
        if target_mean != 'N/A':
            upside = ((target_mean - current_price) / current_price * 100)
            st.metric("Target Price", f"${target_mean:.2f}", delta=f"{upside:+.1f}%")
        
        if target_high != 'N/A':
            st.write(f"**High Target**: ${target_high:.2f}")
        if target_low != 'N/A':
            st.write(f"**Low Target**: ${target_low:.2f}")

def render_industry_comparison(ticker, info, current_price):
    """Render industry comparison section"""
    industry_benchmarks = get_industry_comparison(ticker, info)
    sector = info.get('sector', 'Unknown')
    
    current_metrics = {
        'pe_ratio': info.get('trailingPE', 0),
        'pb_ratio': info.get('priceToBook', 0),
        'roe': 0
    }
    
    # Get ROE from calculated ratios
    if 'income_stmt' in st.session_state and 'balance_sheet' in st.session_state:
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        ratios = calculate_financial_ratios(
            st.session_state['income_stmt'], 
            st.session_state['balance_sheet'], 
            info, 
            current_price, 
            shares_outstanding
        )
        current_metrics['roe'] = ratios.get('ROE %', 0)
    
    industry_fig = create_industry_comparison_chart(current_metrics, industry_benchmarks, sector)
    if industry_fig:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(industry_fig, use_container_width=True)
        with col2:
            render_industry_benchmarks(industry_benchmarks, current_metrics, sector)

def render_industry_benchmarks(industry_benchmarks, current_metrics, sector):
    """Render industry benchmark information"""
    st.markdown(f"**📋 {sector} Industry Benchmarks**")
    st.write(f"**Avg P/E Ratio**: {industry_benchmarks['avg_pe']:.1f}")
    st.write(f"**Avg P/B Ratio**: {industry_benchmarks['avg_pb']:.1f}")
    st.write(f"**Avg ROE**: {industry_benchmarks['avg_roe']:.1f}%")
    
    # Performance vs industry
    st.markdown("**📈 Performance vs Industry**")
    pe_ratio = current_metrics['pe_ratio']
    if pe_ratio and pe_ratio > 0:
        pe_vs_industry = "Above" if pe_ratio > industry_benchmarks['avg_pe'] else "Below"
        pe_color = "🔴" if pe_vs_industry == "Above" else "🟢"
        st.write(f"P/E: {pe_color} {pe_vs_industry} Industry Avg")
    
    roe = current_metrics['roe']
    if roe and roe != 0:
        roe_vs_industry = "Above" if roe > industry_benchmarks['avg_roe'] else "Below"
        roe_color = "🟢" if roe_vs_industry == "Above" else "🔴"
        st.write(f"ROE: {roe_color} {roe_vs_industry} Industry Avg")

def render_technical_analysis(data, ticker):
    """Render technical analysis section"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("📊 Technical Analysis")
    
    show_sma = st.session_state.get('show_sma', False)
    show_bollinger = st.session_state.get('show_bollinger', False)
    
    candlestick_fig = create_candlestick_chart(data, ticker, show_sma, show_bollinger)
    st.plotly_chart(candlestick_fig, use_container_width=True)

def render_financial_performance():
    """Render financial performance section"""
    if 'income_stmt' not in st.session_state:
        return
        
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("💰 Financial Performance")
    
    income_stmt = st.session_state['income_stmt']
    balance_sheet = st.session_state['balance_sheet']
    cash_flow = st.session_state['cash_flow']
    
    # Historical Charts
    charts = create_historical_performance_chart(income_stmt, balance_sheet, cash_flow)
    
    if charts:
        st.subheader("📊 Historical Financial Charts")
        
        # Combined charts
        combined_income_fig = create_combined_income_chart(income_stmt)
        if combined_income_fig:
            st.plotly_chart(combined_income_fig, use_container_width=True)
        
        combined_balance_fig = create_combined_balance_sheet_chart(balance_sheet)
        if combined_balance_fig:
            st.plotly_chart(combined_balance_fig, use_container_width=True)
        
        # Cash flow charts
        cash_flow_charts = [key for key in charts.keys() if key in ['operating_cf', 'free_cf']]
        if cash_flow_charts:
            st.subheader("💸 Cash Flow Analysis")
            if len(cash_flow_charts) == 2:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts[cash_flow_charts[0]], use_container_width=True)
                with col2:
                    st.plotly_chart(charts[cash_flow_charts[1]], use_container_width=True)
    
    # Historical Ratios
    historical_ratios_fig = create_historical_ratios_chart(income_stmt, balance_sheet)
    if historical_ratios_fig:
        st.subheader("📈 Historical Financial Ratios")
        st.plotly_chart(historical_ratios_fig, use_container_width=True)
    
    # Waterfall and Sankey
    render_advanced_financial_charts(income_stmt)

def render_advanced_financial_charts(income_stmt):
    """Render waterfall and sankey charts"""
    if income_stmt is None or income_stmt.empty:
        return
        
    available_years = [str(col)[:4] for col in income_stmt.columns]
    
    # Waterfall Chart
    st.subheader("💧 Income Statement Waterfall Analysis")
    selected_year_waterfall = st.selectbox("Select Year for Waterfall Chart", available_years, key="waterfall_year")
    
    waterfall_fig = create_waterfall_chart(income_stmt, selected_year_waterfall)
    if waterfall_fig:
        st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Sankey Diagram
    st.subheader("🌊 Income Statement Flow (Sankey Diagram)")
    selected_year_sankey = st.selectbox("Select Year for Sankey Diagram", available_years, key="sankey_year")
    
    sankey_fig = create_sankey_diagram(income_stmt, selected_year_sankey)
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)

def render_company_information(info):
    """Render company information section"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("ℹ️ Company Information")
    
    if info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Country:** {info.get('country', 'N/A')}")
        
        with col2:
            employees = info.get('fullTimeEmployees', 'N/A')
            if employees != 'N/A':
                st.write(f"**Full Time Employees:** {employees:,}")
            else:
                st.write("**Full Time Employees:** N/A")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
        
        # Business Summary
        if 'longBusinessSummary' in info:
            st.subheader("Business Summary")
            st.write(info['longBusinessSummary'])

def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    st.info("👆 Enter a stock ticker in the sidebar and click 'Fetch Data' to begin analysis")
    
    # Example tickers
    st.subheader("Popular Stock Tickers")
    example_tickers = {
        "Technology": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
        "Consumer": ["AMZN", "WMT", "HD", "MCD", "NKE"]
    }
    
    for sector, tickers in example_tickers.items():
        st.write(f"**{sector}:** {', '.join(tickers)}")

if __name__ == "__main__":
    main()