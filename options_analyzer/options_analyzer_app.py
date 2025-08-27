import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time

# Set page config
st.set_page_config(page_title="Options Trading Analyzer", layout="wide")

class OptionsDataFetcher:
    """Handles fetching and processing options data"""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_options_data(ticker):
        """Fetch live options data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Validate ticker
            try:
                info = stock.info
                if not info or 'symbol' not in info:
                    return None, None, f"Invalid ticker symbol: {ticker}"
            except:
                return None, None, f"Invalid ticker symbol: {ticker}"
            
            # Get current stock price
            hist = stock.history(period="1d")
            if hist.empty:
                return None, None, f"No price data available for {ticker}"
                
            current_price = hist['Close'].iloc[-1]
            
            # Get options expirations
            try:
                expirations = stock.options
            except:
                return None, None, f"No options data available for {ticker}"
            
            if not expirations:
                return None, None, f"No options expiration dates found for {ticker}"
            
            # Fetch options data
            all_options_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_expirations = len(expirations)
            
            for i, exp_date in enumerate(expirations[:max_expirations]):
                try:
                    status_text.text(f"Fetching {ticker} options data for expiration: {exp_date}")
                    
                    opt_chain = stock.option_chain(exp_date)
                    
                    # Process calls
                    if not opt_chain.calls.empty:
                        calls_data = opt_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].copy()
                        calls_data['contract_date'] = exp_date
                        calls_data['option_type'] = 'call'
                        calls_data = calls_data.rename(columns={'strike': 'strike_price', 'lastPrice': 'option_cost'})
                        calls_data = calls_data[calls_data['option_cost'] > 0.01]
                        if not calls_data.empty:
                            all_options_data.append(calls_data)
                    
                    # Process puts
                    if not opt_chain.puts.empty:
                        puts_data = opt_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].copy()
                        puts_data['contract_date'] = exp_date
                        puts_data['option_type'] = 'put'
                        puts_data = puts_data.rename(columns={'strike': 'strike_price', 'lastPrice': 'option_cost'})
                        puts_data = puts_data[puts_data['option_cost'] > 0.01]
                        if not puts_data.empty:
                            all_options_data.append(puts_data)
                    
                    progress_bar.progress((i + 1) / max_expirations)
                    time.sleep(0.1)
                    
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if all_options_data:
                df = pd.concat(all_options_data, ignore_index=True)
                df['contract_date'] = pd.to_datetime(df['contract_date'])
                df = df.dropna(subset=['option_cost', 'strike_price'])
                df = df[df['option_cost'] > 0]
                return df, current_price, None
            else:
                return None, current_price, f"No valid options data could be retrieved for {ticker}"
                
        except Exception as e:
            return None, None, f"Error fetching data for {ticker}: {str(e)}"

class OptionsAnalyzer:
    """Handles options analysis and calculations"""
    
    @staticmethod
    def calculate_metrics(df, current_price):
        """Calculate additional metrics for options analysis"""
        analysis_df = df.copy()
        current_date = pd.Timestamp.now()
        
        # Basic metrics
        analysis_df['days_to_expiry'] = (analysis_df['contract_date'] - current_date).dt.days
        analysis_df['moneyness'] = analysis_df['strike_price'] / current_price
        analysis_df['distance_from_atm'] = abs(analysis_df['strike_price'] - current_price)
        
        # Intrinsic and time value
        analysis_df['intrinsic_value'] = 0
        analysis_df['time_value'] = 0
        analysis_df['time_value_pct'] = 0
        
        for idx, row in analysis_df.iterrows():
            if row['option_type'] == 'call':
                intrinsic = max(0, current_price - row['strike_price'])
            else:
                intrinsic = max(0, row['strike_price'] - current_price)
            
            analysis_df.loc[idx, 'intrinsic_value'] = intrinsic
            time_val = row['option_cost'] - intrinsic
            analysis_df.loc[idx, 'time_value'] = time_val
            if row['option_cost'] > 0:
                analysis_df.loc[idx, 'time_value_pct'] = (time_val / row['option_cost']) * 100
        
        # Potential returns and breakeven
        analysis_df['potential_return_10pct'] = 0
        analysis_df['potential_return_20pct'] = 0
        analysis_df['breakeven_price'] = 0
        
        for idx, row in analysis_df.iterrows():
            if row['option_type'] == 'call':
                breakeven = row['strike_price'] + row['option_cost']
                new_price_10 = current_price * 1.10
                new_price_20 = current_price * 1.20
                new_value_10 = max(0, new_price_10 - row['strike_price'])
                new_value_20 = max(0, new_price_20 - row['strike_price'])
            else:
                breakeven = row['strike_price'] - row['option_cost']
                new_price_10 = current_price * 0.90
                new_price_20 = current_price * 0.80
                new_value_10 = max(0, row['strike_price'] - new_price_10)
                new_value_20 = max(0, row['strike_price'] - new_price_20)
            
            analysis_df.loc[idx, 'breakeven_price'] = breakeven
            if row['option_cost'] > 0:
                analysis_df.loc[idx, 'potential_return_10pct'] = (new_value_10 - row['option_cost']) / row['option_cost'] * 100
                analysis_df.loc[idx, 'potential_return_20pct'] = (new_value_20 - row['option_cost']) / row['option_cost'] * 100
        
        return analysis_df

class OptionsVisualizer:
    """Handles creating visualizations"""
    
    @staticmethod
    def create_strike_vs_premium_chart(filtered_df, ticker, current_price, option_type):
        """Create strike price vs premium chart"""
        type_data = filtered_df[filtered_df['option_type'] == option_type]
        
        if type_data.empty:
            return None
        
        fig = go.Figure()
        exp_dates = sorted(type_data['contract_date'].unique())
        colors = px.colors.qualitative.Set3
        
        for i, exp_date in enumerate(exp_dates):
            exp_data = type_data[type_data['contract_date'] == exp_date].sort_values('strike_price')
            
            if not exp_data.empty:
                fig.add_trace(go.Scatter(
                    x=exp_data['strike_price'],
                    y=exp_data['option_cost'],
                    mode='lines+markers',
                    name=f"{exp_date.strftime('%Y-%m-%d')}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>Strike: $%{x}<br>Premium: $%{y:.2f}<br>Volume: %{customdata}<extra></extra>',
                    customdata=exp_data['volume']
                ))
        
        fig.add_vline(x=current_price, line_dash="dash", line_color="red", 
                     annotation_text=f"Current: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{ticker} {option_type.upper()} Options - Strike Price vs Premium",
            xaxis_title="Strike Price ($)",
            yaxis_title="Option Premium ($)",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_expiry_vs_premium_chart(filtered_df, ticker, option_type):
        """Create expiry date vs premium chart"""
        type_data = filtered_df[filtered_df['option_type'] == option_type]
        
        if type_data.empty:
            return None
        
        fig = go.Figure()
        strike_prices = sorted(type_data['strike_price'].unique())
        colors = px.colors.qualitative.Plotly
        
        for i, strike in enumerate(strike_prices):
            strike_data = type_data[type_data['strike_price'] == strike].sort_values('contract_date')
            
            if not strike_data.empty:
                fig.add_trace(go.Scatter(
                    x=strike_data['contract_date'],
                    y=strike_data['option_cost'],
                    mode='lines+markers',
                    name=f"Strike ${strike:.1f}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>Strike $%{fullData.name}</b><br>Expiry: %{x}<br>Premium: $%{y:.2f}<br>Volume: %{customdata}<extra></extra>',
                    customdata=strike_data['volume']
                ))
        
        fig.update_layout(
            title=f"{ticker} {option_type.upper()} Options - Expiry Date vs Premium (by Strike)",
            xaxis_title="Expiry Date",
            yaxis_title="Option Premium ($)",
            height=500,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    @staticmethod
    def create_risk_reward_chart(strategy_df, ticker, max_budget):
        """Create risk vs reward scatter plot"""
        fig = px.scatter(
            strategy_df,
            x='option_cost',
            y='potential_return_20pct',
            color='option_type',
            size='volume',
            hover_data=['strike_price', 'contract_date', 'days_to_expiry'],
            title=f"{ticker} Options: Risk (Cost) vs Reward (Potential Return)",
            labels={
                'option_cost': 'Option Cost (Risk) - $',
                'potential_return_20pct': 'Potential Return @ 20% Move (%)'
            }
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break Even")
        fig.add_vline(x=max_budget, line_dash="dash", line_color="orange", annotation_text="Budget Limit")
        
        return fig

def main():
    """Main application function"""
    st.title("📈 Options Trading Analyzer")
    
    # Header controls
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Enter Ticker Symbol:", 
            value="PLTR",
            placeholder="e.g., AAPL, TSLA, SPY, MSFT",
            help="Enter any valid stock ticker symbol"
        ).upper().strip()
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if not ticker_input:
        st.warning("⚠️ Please enter a ticker symbol to continue.")
        st.stop()
    
    # Fetch data
    st.info(f"🔄 Fetching live {ticker_input} options data from Yahoo Finance...")
    df, current_price, error_msg = OptionsDataFetcher.fetch_options_data(ticker_input)
    
    if error_msg:
        st.error(f"❌ {error_msg}")
        st.info("💡 Try a different ticker symbol (e.g., AAPL, MSFT, TSLA, SPY)")
        st.stop()
    
    if df is not None and not df.empty:
        st.success(f"✅ Data fetched successfully! Current {ticker_input} price: ${current_price:.2f}")
        
        # Data overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Total Contracts", len(df))
        with col3:
            st.metric("Expiration Dates", df['contract_date'].nunique())
        with col4:
            st.metric("Strike Range", f"${df['strike_price'].min():.0f} - ${df['strike_price'].max():.0f}")
        
        # Sidebar filters
        st.sidebar.header("📊 Filters & Options")
        
        option_types = st.sidebar.multiselect(
            "Option Type",
            options=df['option_type'].unique(),
            default=df['option_type'].unique()
        )
        
        selected_dates = st.sidebar.multiselect(
            "Select Expiration Dates",
            options=sorted(df['contract_date'].dt.date.unique()),
            default=sorted(df['contract_date'].dt.date.unique())[:8]
        )
        
        # Strike price range with smart defaults
        min_strike = float(df['strike_price'].min())
        max_strike = float(df['strike_price'].max())
        default_range_size = min(10, (max_strike - min_strike) / 4)
        default_min = max(min_strike, current_price - default_range_size)
        default_max = min(max_strike, current_price + default_range_size)
        
        strike_range = st.sidebar.slider(
            "Strike Price Range",
            min_value=min_strike,
            max_value=max_strike,
            value=(default_min, default_max),
            step=0.5
        )
        
        min_volume = st.sidebar.slider(
            "Minimum Volume",
            min_value=0,
            max_value=int(df['volume'].max()) if df['volume'].max() > 0 else 100,
            value=0
        )
        
        # Filter data
        filtered_df = df[
            (df['option_type'].isin(option_types)) &
            (df['contract_date'].dt.date.isin(selected_dates)) &
            (df['strike_price'] >= strike_range[0]) &
            (df['strike_price'] <= strike_range[1]) &
            (df['volume'] >= min_volume)
        ].copy()
        
        if len(filtered_df) > 0:
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Price Charts", 
                "🎯 Strategy Analyzer", 
                "🔥 Volume Analysis", 
                "📊 Greeks", 
                "📋 Data Table"
            ])
            
            # Tab 1: Price Charts
            with tab1:
                st.subheader("Options Prices by Strike and Expiration")
                
                for opt_type in filtered_df['option_type'].unique():
                    # Strike vs Premium chart
                    st.write(f"**{opt_type.upper()}S - Strike Price vs Premium**")
                    fig1 = OptionsVisualizer.create_strike_vs_premium_chart(filtered_df, ticker_input, current_price, opt_type)
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    # Expiry vs Premium chart
                    st.write(f"**{opt_type.upper()}S - Expiry Date vs Premium**")
                    fig2 = OptionsVisualizer.create_expiry_vs_premium_chart(filtered_df, ticker_input, opt_type)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
            
            # Tab 2: Strategy Analyzer
            with tab2:
                st.subheader("🎯 Strategy Analyzer - Find Best Options for Profitable Trades")
                
                # Strategy controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    market_outlook = st.selectbox(
                        "Market Outlook",
                        ["Bullish (Buy Calls)", "Bearish (Buy Puts)", "Neutral (Sell Premium)", "High Volatility (Straddle)"]
                    )
                
                with col2:
                    time_horizon = st.selectbox(
                        "Time Horizon",
                        ["Short-term (< 30 days)", "Medium-term (30-90 days)", "Long-term (> 90 days)", "All timeframes"]
                    )
                
                with col3:
                    max_budget = st.number_input(
                        "Max Budget per Contract ($)",
                        min_value=0.01,
                        max_value=10000.0,
                        value=500.0,
                        step=50.0
                    )
                
                # Calculate analysis metrics
                analysis_df = OptionsAnalyzer.calculate_metrics(filtered_df, current_price)
                
                # Filter by strategy
                strategy_df = analysis_df[analysis_df['option_cost'] <= max_budget].copy()
                
                # Apply time horizon filter
                if time_horizon == "Short-term (< 30 days)":
                    strategy_df = strategy_df[strategy_df['days_to_expiry'] < 30]
                elif time_horizon == "Medium-term (30-90 days)":
                    strategy_df = strategy_df[(strategy_df['days_to_expiry'] >= 30) & (strategy_df['days_to_expiry'] <= 90)]
                elif time_horizon == "Long-term (> 90 days)":
                    strategy_df = strategy_df[strategy_df['days_to_expiry'] > 90]
                
                # Apply market outlook filter
                if "Bullish" in market_outlook:
                    strategy_df = strategy_df[strategy_df['option_type'] == 'call']
                    sort_column = 'potential_return_20pct'
                    st.info("🚀 **Bullish Strategy**: Showing call options ranked by potential 20% upside returns")
                elif "Bearish" in market_outlook:
                    strategy_df = strategy_df[strategy_df['option_type'] == 'put']
                    sort_column = 'potential_return_20pct'
                    st.info("📉 **Bearish Strategy**: Showing put options ranked by potential 20% downside returns")
                elif "Neutral" in market_outlook:
                    sort_column = 'time_value_pct'
                    st.info("⚖️ **Neutral Strategy**: Showing options with highest time value % (good for selling premium)")
                else:
                    sort_column = 'impliedVolatility'
                    st.info("💥 **High Volatility Strategy**: Showing options with highest implied volatility")
                
                if not strategy_df.empty:
                    # Top opportunities
                    st.subheader("🏆 Top Trading Opportunities")
                    
                    top_opportunities = strategy_df.nlargest(10, sort_column)[
                        ['option_type', 'strike_price', 'contract_date', 'option_cost', 'volume', 
                         'days_to_expiry', 'breakeven_price', 'potential_return_10pct', 'potential_return_20pct',
                         'time_value', 'time_value_pct', 'impliedVolatility']
                    ].copy()
                    
                    top_opportunities['contract_date'] = top_opportunities['contract_date'].dt.strftime('%Y-%m-%d')
                    top_opportunities = top_opportunities.round(2)
                    top_opportunities.columns = [
                        'Type', 'Strike', 'Expiry', 'Cost', 'Volume', 'DTE', 'Breakeven', 
                        'Return@+10%', 'Return@+20%', 'Time Value', 'Time Value%', 'IV'
                    ]
                    
                    st.dataframe(top_opportunities, hide_index=True, use_container_width=True)
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Best Opportunity",
                            f"${top_opportunities.iloc[0]['Strike']} {top_opportunities.iloc[0]['Type'].upper()}",
                            f"{top_opportunities.iloc[0]['Return@+20%']:.1f}% potential return"
                        )
                    with col2:
                        st.metric("Average Cost", f"${top_opportunities['Cost'].mean():.2f}")
                    with col3:
                        st.metric("Average DTE", f"{top_opportunities['DTE'].mean():.0f} days")
                    
                    # Risk/Reward chart
                    st.subheader("📊 Risk vs Reward Analysis")
                    fig_risk = OptionsVisualizer.create_risk_reward_chart(strategy_df, ticker_input, max_budget)
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Breakeven analysis
                    st.subheader("🎯 Breakeven Analysis")
                    st.write(f"**Current {ticker_input} Price: ${current_price:.2f}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'call' in strategy_df['option_type'].values:
                            call_breakevens = strategy_df[strategy_df['option_type'] == 'call']['breakeven_price']
                            min_call_breakeven = call_breakevens.min()
                            move_needed = ((min_call_breakeven - current_price) / current_price) * 100
                            st.info(f"**CALLS Breakeven Range**\n${min_call_breakeven:.2f} - ${call_breakevens.max():.2f}\nMinimum move needed: **{move_needed:.1f}%** upward")
                    
                    with col2:
                        if 'put' in strategy_df['option_type'].values:
                            put_breakevens = strategy_df[strategy_df['option_type'] == 'put']['breakeven_price']
                            max_put_breakeven = put_breakevens.max()
                            move_needed = ((current_price - max_put_breakeven) / current_price) * 100
                            st.info(f"**PUTS Breakeven Range**\n${put_breakevens.min():.2f} - ${max_put_breakeven:.2f}\nMinimum move needed: **{move_needed:.1f}%** downward")
                
                else:
                    st.warning("⚠️ No options match your criteria. Try adjusting your filters.")
            
            # Tab 3: Volume Analysis
            with tab3:
                st.subheader("Volume Analysis")
                
                vol_by_strike = filtered_df.groupby(['strike_price', 'option_type'])['volume'].sum().reset_index()
                
                fig_vol = px.bar(
                    vol_by_strike, 
                    x='strike_price', 
                    y='volume', 
                    color='option_type',
                    title=f"{ticker_input} Trading Volume by Strike Price",
                    labels={'volume': 'Total Volume', 'strike_price': 'Strike Price ($)'}
                )
                fig_vol.add_vline(x=current_price, line_dash="dash", line_color="red")
                st.plotly_chart(fig_vol, use_container_width=True)
                
                st.subheader("Most Active Contracts")
                top_volume = filtered_df.nlargest(10, 'volume')[
                    ['contract_date', 'option_type', 'strike_price', 'option_cost', 'volume', 'bid', 'ask']
                ].copy()
                top_volume['contract_date'] = top_volume['contract_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(top_volume, hide_index=True)
            
            # Tab 4: Greeks
            with tab4:
                st.subheader("Options Greeks & Metrics")
                
                if 'impliedVolatility' in filtered_df.columns:
                    fig_iv = px.scatter(
                        filtered_df,
                        x='strike_price',
                        y='impliedVolatility',
                        color='option_type',
                        size='volume',
                        hover_data=['contract_date', 'option_cost'],
                        title=f"{ticker_input} Implied Volatility by Strike Price"
                    )
                    fig_iv.add_vline(x=current_price, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_iv, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Call Options Summary**")
                        calls_data = filtered_df[filtered_df['option_type'] == 'call']
                        if not calls_data.empty:
                            calls_summary = calls_data[['option_cost', 'volume', 'impliedVolatility']].describe().round(4)
                            st.dataframe(calls_summary)
                        else:
                            st.info("No call options in filtered data")
                    
                    with col2:
                        st.write("**Put Options Summary**")
                        puts_data = filtered_df[filtered_df['option_type'] == 'put']
                        if not puts_data.empty:
                            puts_summary = puts_data[['option_cost', 'volume', 'impliedVolatility']].describe().round(4)
                            st.dataframe(puts_summary)
                        else:
                            st.info("No put options in filtered data")
            
            # Tab 5: Data Table
            with tab5:
                st.subheader("Complete Options Data")
                
                display_df = filtered_df.copy()
                display_df['contract_date'] = display_df['contract_date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.round(4)
                
                search_term = st.text_input("Search in data:", placeholder="Enter strike price, date, etc.")
                if search_term:
                    display_df = display_df[
                        display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                    ]
                
                st.dataframe(display_df, hide_index=True, use_container_width=True)
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{ticker_input.lower()}_options_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.warning("⚠️ No data matches your current filters. Please adjust the filter settings.")
    
    else:
        st.error("❌ No options data available for this ticker.")
    
    # Info section
    with st.expander("ℹ️ About This App"):
        st.markdown(f"""
        ### 📊 Live Options Trading Analyzer for {ticker_input}
        
        **Features:**
        - 🔄 Live data fetching for any ticker symbol
        - 📈 Multiple chart types for price analysis
        - 🎯 **Strategy Analyzer** with profit/loss calculations
        - 📊 Volume analysis and Greeks
        - 🔍 Advanced filtering and search capabilities
        
        **Strategy Analyzer Includes:**
        - Breakeven price calculations
        - Potential return scenarios (10% & 20% moves)
        - Risk vs reward visualization
        - Time value analysis
        - Trading recommendations
        
        **Tips:**
        - Try popular tickers: AAPL, MSFT, TSLA, SPY, QQQ
        - Use the Strategy Analyzer to find profitable trades
        - Filter by volume for liquid options
        - Check breakeven analysis before trading
        """)

if __name__ == "__main__":
    main()