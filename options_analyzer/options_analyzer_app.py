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
            
            max_expirations = min(len(expirations), 15)
            
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
        
        fig.add_vline(x=current_price, line_dash="dash", line_color="red") #annotation_text=f"Current: ${current_price:.2f}")
        
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
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Price Charts", 
                "🎯 Strategy Analyzer", 
                "🔥 Volume Analysis", 
                "🌊 Options Flow", 
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
                
                # Chart 1: Volume by Strike Price (existing)
                st.write("**📊 Trading Volume by Strike Price**")
                vol_by_strike = filtered_df.groupby(['strike_price', 'option_type'])['volume'].sum().reset_index()
                
                fig_vol_strike = px.bar(
                    vol_by_strike, 
                    x='strike_price', 
                    y='volume', 
                    color='option_type',
                    title=f"{ticker_input} Trading Volume by Strike Price",
                    labels={'volume': 'Total Volume', 'strike_price': 'Strike Price ($)'},
                    color_discrete_map={'call': '#2E8B57', 'put': '#DC143C'}
                )
                fig_vol_strike.add_vline(x=current_price, line_dash="dash", line_color="red") #annotation_text=f"Current: ${current_price:.2f}")
                st.plotly_chart(fig_vol_strike, use_container_width=True)
                
                # Chart 2: Volume by Expiry Date (NEW)
                st.write("**📅 Trading Volume by Expiry Date**")
                vol_by_expiry = filtered_df.groupby([filtered_df['contract_date'].dt.date, 'option_type'])['volume'].sum().reset_index()
                vol_by_expiry['contract_date'] = pd.to_datetime(vol_by_expiry['contract_date'])
                
                fig_vol_expiry = px.bar(
                    vol_by_expiry, 
                    x='contract_date', 
                    y='volume', 
                    color='option_type',
                    title=f"{ticker_input} Trading Volume by Expiry Date",
                    labels={'volume': 'Total Volume', 'contract_date': 'Expiry Date'},
                    color_discrete_map={'call': '#2E8B57', 'put': '#DC143C'}
                )
                
                # Add annotations for high volume dates
                if not vol_by_expiry.empty:
                    max_volume_date = vol_by_expiry.loc[vol_by_expiry['volume'].idxmax(), 'contract_date']
                    fig_vol_expiry.add_vline(x=max_volume_date, line_dash="dot", line_color="orange") #annotation_text="Highest Volume")
                
                fig_vol_expiry.update_xaxes(tickangle=45)
                st.plotly_chart(fig_vol_expiry, use_container_width=True)
                
                # Volume insights
                col1, col2, col3 = st.columns(3)
                
                total_call_volume = filtered_df[filtered_df['option_type'] == 'call']['volume'].sum()
                total_put_volume = filtered_df[filtered_df['option_type'] == 'put']['volume'].sum()
                put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
                
                with col1:
                    st.metric("Total Call Volume", f"{total_call_volume:,.0f}")
                with col2:
                    st.metric("Total Put Volume", f"{total_put_volume:,.0f}")
                with col3:
                    st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
                
                # Volume analysis insights
                st.subheader("📈 Volume Analysis Insights")
                
                # Find most active expiry
                most_active_expiry = vol_by_expiry.groupby('contract_date')['volume'].sum().idxmax()
                most_active_volume = vol_by_expiry.groupby('contract_date')['volume'].sum().max()
                
                # Find most active strike
                most_active_strike = vol_by_strike.groupby('strike_price')['volume'].sum().idxmax()
                most_active_strike_volume = vol_by_strike.groupby('strike_price')['volume'].sum().max()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **🎯 Most Active Expiry Date**  
                    **{most_active_expiry.strftime('%Y-%m-%d')}**  
                    Total Volume: {most_active_volume:,.0f}  
                    Days to Expiry: {(most_active_expiry - pd.Timestamp.now()).days} days
                    """)
                
                with col2:
                    moneyness = most_active_strike / current_price
                    strike_type = "ITM" if moneyness < 1 else "ATM" if abs(moneyness - 1) < 0.02 else "OTM"
                    st.info(f"""
                    **🎯 Most Active Strike Price**  
                    **${most_active_strike:.2f}** ({strike_type})  
                    Total Volume: {most_active_strike_volume:,.0f}  
                    Distance from Current: {((most_active_strike - current_price) / current_price * 100):+.1f}%
                    """)
                
                # Weekly vs Monthly expiry analysis
                weekly_monthly_analysis = filtered_df.copy()
                weekly_monthly_analysis['days_to_expiry'] = (weekly_monthly_analysis['contract_date'] - pd.Timestamp.now()).dt.days
                weekly_monthly_analysis['expiry_type'] = weekly_monthly_analysis['days_to_expiry'].apply(
                    lambda x: 'Weekly (0-14 days)' if x <= 14 else 'Monthly (15-45 days)' if x <= 45 else 'Quarterly (45+ days)'
                )
                
                expiry_type_volume = weekly_monthly_analysis.groupby(['expiry_type', 'option_type'])['volume'].sum().reset_index()
                
                if not expiry_type_volume.empty:
                    st.write("**⏰ Volume Distribution by Expiry Type**")
                    fig_expiry_type = px.bar(
                        expiry_type_volume,
                        x='expiry_type',
                        y='volume',
                        color='option_type',
                        title="Volume Distribution: Weekly vs Monthly vs Quarterly Options",
                        labels={'volume': 'Total Volume', 'expiry_type': 'Expiry Type'},
                        color_discrete_map={'call': '#2E8B57', 'put': '#DC143C'}
                    )
                    st.plotly_chart(fig_expiry_type, use_container_width=True)
                
                st.subheader("Most Active Contracts")
                top_volume = filtered_df.nlargest(10, 'volume')[
                    ['contract_date', 'option_type', 'strike_price', 'option_cost', 'volume', 'bid', 'ask']
                ].copy()
                top_volume['contract_date'] = top_volume['contract_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(top_volume, hide_index=True)
            
            # Tab 4: Options Flow Analysis (NEW)
            with tab4:
                st.subheader("🌊 Options Flow Analysis")
                st.info("📊 **Options Flow Analysis** helps identify unusual activity, sentiment shifts, and institutional money movement")
                
                # Calculate flow metrics
                flow_df = filtered_df.copy()
                flow_df['days_to_expiry'] = (flow_df['contract_date'] - pd.Timestamp.now()).dt.days
                flow_df['moneyness'] = flow_df['strike_price'] / current_price
                flow_df['dollar_volume'] = flow_df['volume'] * flow_df['option_cost'] * 100  # *100 for contract multiplier
                
                # Classify moneyness
                def classify_moneyness(moneyness, option_type):
                    if option_type == 'call':
                        if moneyness < 0.95:
                            return 'Deep ITM'
                        elif moneyness < 0.98:
                            return 'ITM'
                        elif moneyness <= 1.02:
                            return 'ATM'
                        elif moneyness <= 1.05:
                            return 'OTM'
                        else:
                            return 'Deep OTM'
                    else:  # put
                        if moneyness > 1.05:
                            return 'Deep ITM'
                        elif moneyness > 1.02:
                            return 'ITM'
                        elif moneyness >= 0.98:
                            return 'ATM'
                        elif moneyness >= 0.95:
                            return 'OTM'
                        else:
                            return 'Deep OTM'
                
                flow_df['moneyness_class'] = flow_df.apply(lambda row: classify_moneyness(row['moneyness'], row['option_type']), axis=1)
                
                # Flow Analysis Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_call_dollar_vol = flow_df[flow_df['option_type'] == 'call']['dollar_volume'].sum()
                total_put_dollar_vol = flow_df[flow_df['option_type'] == 'put']['dollar_volume'].sum()
                total_dollar_volume = total_call_dollar_vol + total_put_dollar_vol
                
                call_volume = flow_df[flow_df['option_type'] == 'call']['volume'].sum()
                put_volume = flow_df[flow_df['option_type'] == 'put']['volume'].sum()
                pc_ratio = put_volume / call_volume if call_volume > 0 else 0
                
                with col1:
                    st.metric("Total $ Volume", f"${total_dollar_volume/1e6:.1f}M")
                with col2:
                    st.metric("Put/Call Ratio", f"{pc_ratio:.2f}", 
                             delta="Bearish" if pc_ratio > 1 else "Bullish")
                with col3:
                    call_percentage = (total_call_dollar_vol / total_dollar_volume * 100) if total_dollar_volume > 0 else 0
                    st.metric("Call $ Flow %", f"{call_percentage:.1f}%")
                with col4:
                    put_percentage = (total_put_dollar_vol / total_dollar_volume * 100) if total_dollar_volume > 0 else 0
                    st.metric("Put $ Flow %", f"{put_percentage:.1f}%")
                
                # Flow Analysis Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**💰 Dollar Volume Flow by Moneyness**")
                    
                    # Dollar volume by moneyness
                    dollar_vol_by_moneyness = flow_df.groupby(['moneyness_class', 'option_type'])['dollar_volume'].sum().reset_index()
                    dollar_vol_by_moneyness['dollar_volume_millions'] = dollar_vol_by_moneyness['dollar_volume'] / 1e6
                    
                    # Define order for moneyness classes
                    moneyness_order = ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Deep OTM']
                    dollar_vol_by_moneyness['moneyness_class'] = pd.Categorical(
                        dollar_vol_by_moneyness['moneyness_class'], 
                        categories=moneyness_order, 
                        ordered=True
                    )
                    dollar_vol_by_moneyness = dollar_vol_by_moneyness.sort_values('moneyness_class')
                    
                    fig_dollar_flow = px.bar(
                        dollar_vol_by_moneyness,
                        x='moneyness_class',
                        y='dollar_volume_millions',
                        color='option_type',
                        title=f"{ticker_input} Dollar Volume Flow by Moneyness",
                        labels={'dollar_volume_millions': 'Dollar Volume ($M)', 'moneyness_class': 'Moneyness'},
                        color_discrete_map={'call': '#2E8B57', 'put': '#DC143C'}
                    )
                    st.plotly_chart(fig_dollar_flow, use_container_width=True)
                
                with col2:
                    st.write("**⏰ Flow by Time to Expiry**")
                    
                    # Classify by time to expiry
                    def classify_dte(days):
                        if days <= 7:
                            return '0-7 days'
                        elif days <= 30:
                            return '8-30 days'
                        elif days <= 90:
                            return '31-90 days'
                        else:
                            return '90+ days'
                    
                    flow_df['dte_class'] = flow_df['days_to_expiry'].apply(classify_dte)
                    
                    dollar_vol_by_dte = flow_df.groupby(['dte_class', 'option_type'])['dollar_volume'].sum().reset_index()
                    dollar_vol_by_dte['dollar_volume_millions'] = dollar_vol_by_dte['dollar_volume'] / 1e6
                    
                    # Order DTE classes
                    dte_order = ['0-7 days', '8-30 days', '31-90 days', '90+ days']
                    dollar_vol_by_dte['dte_class'] = pd.Categorical(
                        dollar_vol_by_dte['dte_class'], 
                        categories=dte_order, 
                        ordered=True
                    )
                    dollar_vol_by_dte = dollar_vol_by_dte.sort_values('dte_class')
                    
                    fig_dte_flow = px.bar(
                        dollar_vol_by_dte,
                        x='dte_class',
                        y='dollar_volume_millions',
                        color='option_type',
                        title=f"{ticker_input} Dollar Volume Flow by Time to Expiry",
                        labels={'dollar_volume_millions': 'Dollar Volume ($M)', 'dte_class': 'Days to Expiry'},
                        color_discrete_map={'call': '#2E8B57', 'put': '#DC143C'}
                    )
                    st.plotly_chart(fig_dte_flow, use_container_width=True)
                
                # Unusual Activity Detection
                st.subheader("🚨 Unusual Activity Detection")
                
                # Calculate unusual activity metrics
                flow_df['volume_rank'] = flow_df['volume'].rank(pct=True)
                flow_df['dollar_volume_rank'] = flow_df['dollar_volume'].rank(pct=True)
                flow_df['iv_rank'] = flow_df['impliedVolatility'].rank(pct=True) if 'impliedVolatility' in flow_df.columns else 0
                
                # Define unusual activity (top 10% in volume or dollar volume)
                unusual_activity = flow_df[
                    (flow_df['volume_rank'] >= 0.9) | 
                    (flow_df['dollar_volume_rank'] >= 0.9)
                ].copy()
                
                if not unusual_activity.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🔥 High Volume Contracts**")
                        high_volume = unusual_activity.nlargest(5, 'volume')[
                            ['option_type', 'strike_price', 'contract_date', 'volume', 'option_cost', 'dollar_volume']
                        ].copy()
                        high_volume['contract_date'] = high_volume['contract_date'].dt.strftime('%Y-%m-%d')
                        high_volume['dollar_volume'] = (high_volume['dollar_volume'] / 1000).round(0).astype(int)
                        high_volume.columns = ['Type', 'Strike', 'Expiry', 'Volume', 'Cost', 'Dollar Vol (K)']
                        st.dataframe(high_volume, hide_index=True)
                    
                    with col2:
                        st.write("**💰 High Dollar Volume Contracts**")
                        high_dollar = unusual_activity.nlargest(5, 'dollar_volume')[
                            ['option_type', 'strike_price', 'contract_date', 'volume', 'option_cost', 'dollar_volume']
                        ].copy()
                        high_dollar['contract_date'] = high_dollar['contract_date'].dt.strftime('%Y-%m-%d')
                        high_dollar['dollar_volume'] = (high_dollar['dollar_volume'] / 1000).round(0).astype(int)
                        high_dollar.columns = ['Type', 'Strike', 'Expiry', 'Volume', 'Cost', 'Dollar Vol (K)']
                        st.dataframe(high_dollar, hide_index=True)
                else:
                    st.info("No unusual activity detected in current filtered data")
                
                # Flow Sentiment Analysis
                st.subheader("📈 Flow Sentiment Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                # ATM flow analysis
                atm_flow = flow_df[flow_df['moneyness_class'] == 'ATM']
                atm_call_vol = atm_flow[atm_flow['option_type'] == 'call']['dollar_volume'].sum()
                atm_put_vol = atm_flow[atm_flow['option_type'] == 'put']['dollar_volume'].sum()
                atm_ratio = atm_put_vol / atm_call_vol if atm_call_vol > 0 else 0
                
                with col1:
                    sentiment = "🐻 Bearish" if atm_ratio > 1.2 else "🐂 Bullish" if atm_ratio < 0.8 else "⚖️ Neutral"
                    st.metric("ATM Flow Sentiment", sentiment, f"P/C: {atm_ratio:.2f}")
                
                # Short-term flow (0-30 days)
                short_term_flow = flow_df[flow_df['days_to_expiry'] <= 30]
                st_call_vol = short_term_flow[short_term_flow['option_type'] == 'call']['dollar_volume'].sum()
                st_put_vol = short_term_flow[short_term_flow['option_type'] == 'put']['dollar_volume'].sum()
                st_ratio = st_put_vol / st_call_vol if st_call_vol > 0 else 0
                
                with col2:
                    st_sentiment = "🐻 Bearish" if st_ratio > 1.2 else "🐂 Bullish" if st_ratio < 0.8 else "⚖️ Neutral"
                    st.metric("Short-term Sentiment", st_sentiment, f"P/C: {st_ratio:.2f}")
                
                # Large trades (top 25% by dollar volume)
                large_trades = flow_df[flow_df['dollar_volume_rank'] >= 0.75]
                lt_call_vol = large_trades[large_trades['option_type'] == 'call']['dollar_volume'].sum()
                lt_put_vol = large_trades[large_trades['option_type'] == 'put']['dollar_volume'].sum()
                lt_ratio = lt_put_vol / lt_call_vol if lt_call_vol > 0 else 0
                
                with col3:
                    lt_sentiment = "🐻 Bearish" if lt_ratio > 1.2 else "🐂 Bullish" if lt_ratio < 0.8 else "⚖️ Neutral"
                    st.metric("Large Trade Sentiment", lt_sentiment, f"P/C: {lt_ratio:.2f}")
                
                # Flow Summary Insights
                st.subheader("💡 Flow Analysis Insights")
                
                # Find dominant flows
                dominant_moneyness = dollar_vol_by_moneyness.loc[dollar_vol_by_moneyness['dollar_volume_millions'].idxmax(), 'moneyness_class']
                dominant_dte = dollar_vol_by_dte.loc[dollar_vol_by_dte['dollar_volume_millions'].idxmax(), 'dte_class']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"""
                    **🎯 Dominant Flow Patterns**
                    - **Moneyness**: {dominant_moneyness} options seeing most $ flow
                    - **Time Frame**: {dominant_dte} options most active
                    - **Overall Bias**: {'Bullish' if pc_ratio < 1 else 'Bearish'} (P/C: {pc_ratio:.2f})
                    """)
                
                with col2:
                    # Calculate some interesting metrics
                    otm_call_flow = flow_df[(flow_df['option_type'] == 'call') & (flow_df['moneyness_class'].isin(['OTM', 'Deep OTM']))]['dollar_volume'].sum()
                    otm_put_flow = flow_df[(flow_df['option_type'] == 'put') & (flow_df['moneyness_class'].isin(['OTM', 'Deep OTM']))]['dollar_volume'].sum()
                    
                    weekly_flow = flow_df[flow_df['days_to_expiry'] <= 7]['dollar_volume'].sum()
                    monthly_flow = flow_df[(flow_df['days_to_expiry'] > 7) & (flow_df['days_to_expiry'] <= 30)]['dollar_volume'].sum()
                    
                    st.info(f"""
                    **📊 Key Flow Metrics**
                    - **OTM Calls**: ${otm_call_flow/1e6:.1f}M (speculation/hedging)
                    - **OTM Puts**: ${otm_put_flow/1e6:.1f}M (protection/bearish)
                    - **Weekly Flow**: ${weekly_flow/1e6:.1f}M vs Monthly: ${monthly_flow/1e6:.1f}M
                    """)
            
            # Tab 5: Greeks (renumbered)
            with tab5:
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
            
            # Tab 6: Data Table (renumbered)
            with tab6:
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
