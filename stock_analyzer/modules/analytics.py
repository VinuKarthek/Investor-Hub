"""
Analytics Module
Handles financial calculations, ratios, and advanced analytics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

def calculate_financial_ratios(income_stmt, balance_sheet, info, current_price, shares_outstanding):
    """
    Calculate comprehensive financial ratios
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        info (dict): Stock info from yfinance
        current_price (float): Current stock price
        shares_outstanding (int): Number of shares outstanding
        
    Returns:
        dict: Dictionary of calculated ratios
    """
    ratios = {}
    
    try:
        # Get latest year data
        if income_stmt is not None and not income_stmt.empty:
            latest_income = income_stmt.iloc[:, 0]  # Most recent year
        else:
            latest_income = pd.Series()
            
        if balance_sheet is not None and not balance_sheet.empty:
            latest_balance = balance_sheet.iloc[:, 0]  # Most recent year
        else:
            latest_balance = pd.Series()
        
        # Profitability Ratios
        revenue = latest_income.get('Total Revenue', 0)
        net_income = latest_income.get('Net Income', 0)
        gross_profit = latest_income.get('Gross Profit', 0)
        operating_income = latest_income.get('Operating Income', 0)
        
        if revenue > 0:
            ratios['Gross Margin %'] = (gross_profit / revenue * 100) if gross_profit else 0
            ratios['Operating Margin %'] = (operating_income / revenue * 100) if operating_income else 0
            ratios['Net Margin %'] = (net_income / revenue * 100)
        
        # Liquidity Ratios
        current_assets = latest_balance.get('Current Assets', 0)
        current_liabilities = latest_balance.get('Current Liabilities', 0)
        cash = latest_balance.get('Cash And Cash Equivalents', 0)
        inventory = latest_balance.get('Inventory', 0)
        
        if current_liabilities > 0:
            ratios['Current Ratio'] = current_assets / current_liabilities
            ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities
            ratios['Cash Ratio'] = cash / current_liabilities
        
        # Leverage Ratios
        total_debt = latest_balance.get('Total Debt', 0)
        total_equity = latest_balance.get('Stockholders Equity', 0) or latest_balance.get('Total Stockholder Equity', 0)
        total_assets = latest_balance.get('Total Assets', 0)
        
        if total_equity > 0:
            ratios['Debt to Equity'] = total_debt / total_equity
        if total_assets > 0:
            ratios['Debt to Assets'] = total_debt / total_assets
            ratios['ROA %'] = (net_income / total_assets * 100)
        if total_equity > 0:
            ratios['ROE %'] = (net_income / total_equity * 100)
        
        # Efficiency Ratios
        if revenue > 0 and total_assets > 0:
            ratios['Asset Turnover'] = revenue / total_assets
        
        # Market Ratios (from info)
        market_cap = info.get('marketCap', 0)
        if market_cap > 0 and revenue > 0:
            ratios['P/S Ratio'] = market_cap / revenue
        if market_cap > 0 and net_income > 0:
            ratios['P/E Ratio'] = market_cap / net_income
        
        # Book Value per Share
        if shares_outstanding and total_equity > 0:
            book_value_per_share = total_equity / shares_outstanding
            ratios['Book Value/Share'] = book_value_per_share
            if current_price > 0:
                ratios['P/B Ratio'] = current_price / book_value_per_share
        
        # Additional ratios
        ebitda = latest_income.get('EBITDA', 0)
        if ebitda > 0 and market_cap > 0:
            ratios['EV/EBITDA'] = market_cap / ebitda  # Simplified, should include net debt
    
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
    
    return ratios

def create_eps_trend_chart(income_stmt, info):
    """
    Create EPS trend chart
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        info (dict): Stock info from yfinance
        
    Returns:
        plotly.graph_objects.Figure: EPS trend chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
            
        # Get shares outstanding data
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        
        if not shares_outstanding:
            return None
            
        # Calculate EPS for each year
        net_income_data = income_stmt.loc['Net Income'].dropna() if 'Net Income' in income_stmt.index else pd.Series()
        
        if net_income_data.empty:
            return None
            
        years = [str(col)[:4] for col in net_income_data.index]
        eps_values = [income / shares_outstanding for income in net_income_data.values]
        
        fig = go.Figure()
        
        # Add EPS line
        fig.add_trace(go.Scatter(
            x=years,
            y=eps_values,
            mode='lines+markers',
            name='EPS',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB')
        ))
        
        # Add trend line
        if len(eps_values) > 1:
            z = np.polyfit(range(len(eps_values)), eps_values, 1)
            trend_line = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=years,
                y=trend_line(range(len(eps_values))),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title='Earnings Per Share (EPS) Trend',
            xaxis_title='Year',
            yaxis_title='EPS ($)',
            height=400,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating EPS chart: {str(e)}")
        return None

def create_dividend_history_chart(ticker):
    """
    Create dividend history visualization
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Dividend history chart
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        
        if dividends.empty:
            return None
            
        # Group by year and sum
        dividend_yearly = dividends.groupby(dividends.index.year).sum()
        
        if len(dividend_yearly) < 2:
            return None
            
        fig = go.Figure()
        
        # Add dividend bars
        fig.add_trace(go.Bar(
            x=dividend_yearly.index,
            y=dividend_yearly.values,
            name='Annual Dividends',
            marker_color='#2E8B57',
            width=0.6
        ))
        
        # Add trend line
        if len(dividend_yearly) > 2:
            z = np.polyfit(range(len(dividend_yearly)), dividend_yearly.values, 1)
            trend_line = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=dividend_yearly.index,
                y=trend_line(range(len(dividend_yearly))),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Annual Dividend History',
            xaxis_title='Year',
            yaxis_title='Dividend per Share ($)',
            height=400,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return None

def display_analyst_ratings(recommendations, analyst_price_targets, current_price):
    """
    Display analyst recommendations and price targets
    
    Args:
        recommendations (pd.DataFrame): Analyst recommendations
        analyst_price_targets (dict): Price target data
        current_price (float): Current stock price
        
    Returns:
        tuple: (rating_chart, recommendation_counts)
    """
    try:
        if recommendations is not None and not recommendations.empty:
            # Get most recent recommendations
            recent_recs = recommendations.head(20)  # Last 20 recommendations
            
            # Count recommendations
            rec_counts = recent_recs['To Grade'].value_counts()
            
            # Create rating distribution
            rating_colors = {
                'Buy': '#2E8B57',
                'Strong Buy': '#006400',
                'Outperform': '#32CD32',
                'Hold': '#FFD700',
                'Neutral': '#FFA500',
                'Underperform': '#FF6347',
                'Sell': '#DC143C',
                'Strong Sell': '#8B0000'
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=rec_counts.index,
                values=rec_counts.values,
                hole=.3,
                marker=dict(colors=[rating_colors.get(label, '#808080') for label in rec_counts.index])
            )])
            
            fig.update_layout(
                title="Analyst Recommendations Distribution",
                height=400,
                showlegend=True,
                template='plotly_white'
            )
            
            return fig, rec_counts
        
        return None, None
        
    except Exception as e:
        return None, None

def get_industry_comparison(ticker, info):
    """
    Get basic industry comparison metrics
    
    Args:
        ticker (str): Stock ticker symbol
        info (dict): Stock info from yfinance
        
    Returns:
        dict: Industry benchmark data
    """
    try:
        industry = info.get('industry', '')
        sector = info.get('sector', '')
        
        # Industry averages (these would typically come from a financial data API)
        # For demo purposes, showing typical ranges
        industry_benchmarks = {
            'Technology': {'avg_pe': 25, 'avg_pb': 4, 'avg_roe': 15, 'avg_debt_equity': 0.3},
            'Healthcare': {'avg_pe': 20, 'avg_pb': 3, 'avg_roe': 12, 'avg_debt_equity': 0.4},
            'Financial Services': {'avg_pe': 12, 'avg_pb': 1.2, 'avg_roe': 10, 'avg_debt_equity': 0.8},
            'Consumer Cyclical': {'avg_pe': 18, 'avg_pb': 2.5, 'avg_roe': 14, 'avg_debt_equity': 0.5},
            'Industrials': {'avg_pe': 16, 'avg_pb': 2, 'avg_roe': 11, 'avg_debt_equity': 0.6},
            'Energy': {'avg_pe': 15, 'avg_pb': 1.5, 'avg_roe': 8, 'avg_debt_equity': 0.7},
            'Utilities': {'avg_pe': 18, 'avg_pb': 1.3, 'avg_roe': 9, 'avg_debt_equity': 1.2},
            'Communication Services': {'avg_pe': 22, 'avg_pb': 3, 'avg_roe': 13, 'avg_debt_equity': 0.4}
        }
        
        return industry_benchmarks.get(sector, {'avg_pe': 18, 'avg_pb': 2.5, 'avg_roe': 12, 'avg_debt_equity': 0.5})
        
    except Exception as e:
        return {'avg_pe': 18, 'avg_pb': 2.5, 'avg_roe': 12, 'avg_debt_equity': 0.5}

def create_industry_comparison_chart(current_metrics, industry_benchmarks, sector):
    """
    Create industry comparison chart
    
    Args:
        current_metrics (dict): Current stock metrics
        industry_benchmarks (dict): Industry benchmark data
        sector (str): Industry sector
        
    Returns:
        plotly.graph_objects.Figure: Industry comparison chart
    """
    try:
        metrics = ['P/E Ratio', 'P/B Ratio', 'ROE %']
        current_values = [
            current_metrics.get('pe_ratio', 0),
            current_metrics.get('pb_ratio', 0),
            current_metrics.get('roe', 0)
        ]
        industry_values = [
            industry_benchmarks.get('avg_pe', 0),
            industry_benchmarks.get('avg_pb', 0),
            industry_benchmarks.get('avg_roe', 0)
        ]
        
        # Filter out zero values
        valid_indices = [i for i, (curr, ind) in enumerate(zip(current_values, industry_values)) 
                        if curr != 0 and ind != 0]
        
        if not valid_indices:
            return None
            
        metrics = [metrics[i] for i in valid_indices]
        current_values = [current_values[i] for i in valid_indices]
        industry_values = [industry_values[i] for i in valid_indices]
        
        fig = go.Figure()
        
        # Add current company bars
        fig.add_trace(go.Bar(
            x=metrics,
            y=current_values,
            name='Current Stock',
            marker_color='#2E86AB',
            width=0.35,
            offsetgroup=1
        ))
        
        # Add industry average bars
        fig.add_trace(go.Bar(
            x=metrics,
            y=industry_values,
            name=f'{sector} Industry Avg',
            marker_color='#FFA500',
            width=0.35,
            offsetgroup=2
        ))
        
        fig.update_layout(
            title=f'Stock vs {sector} Industry Comparison',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        return None

def calculate_dcf_valuation(income_stmt, cash_flow, info, growth_rate=0.05, discount_rate=0.10):
    """
    Simple DCF valuation model
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        cash_flow (pd.DataFrame): Cash flow data
        info (dict): Stock info
        growth_rate (float): Assumed growth rate
        discount_rate (float): Discount rate (WACC)
        
    Returns:
        dict: DCF valuation results
    """
    try:
        if cash_flow is None or cash_flow.empty:
            return None
            
        # Get latest free cash flow
        ocf_metrics = ['Operating Cash Flow', 'Total Cash From Operating Activities']
        latest_ocf = 0
        latest_capex = 0
        
        for metric in ocf_metrics:
            if metric in cash_flow.index:
                latest_ocf = cash_flow.loc[metric].iloc[0]
                break
                
        if 'Capital Expenditures' in cash_flow.index:
            latest_capex = cash_flow.loc['Capital Expenditures'].iloc[0]
            
        latest_fcf = latest_ocf + latest_capex  # CapEx is negative
        
        if latest_fcf <= 0:
            return None
            
        # Project 5 years of FCF
        projected_fcf = []
        for year in range(1, 6):
            fcf = latest_fcf * ((1 + growth_rate) ** year)
            projected_fcf.append(fcf)
            
        # Terminal value
        terminal_fcf = projected_fcf[-1] * (1 + growth_rate)
        terminal_value = terminal_fcf / (discount_rate - growth_rate)
        
        # Discount to present value
        pv_fcf = sum([fcf / ((1 + discount_rate) ** year) for year, fcf in enumerate(projected_fcf, 1)])
        pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
        
        enterprise_value = pv_fcf + pv_terminal
        
        # Get shares outstanding
        shares = info.get('sharesOutstanding', 0)
        if shares == 0:
            return None
            
        fair_value_per_share = enterprise_value / shares
        
        return {
            'fair_value': fair_value_per_share,
            'enterprise_value': enterprise_value,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal,
            'latest_fcf': latest_fcf
        }
        
    except Exception as e:
        return None

def calculate_growth_metrics(income_stmt):
    """
    Calculate growth metrics over time
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        dict: Growth metrics
    """
    try:
        if income_stmt is None or income_stmt.empty or len(income_stmt.columns) < 2:
            return None
            
        growth_metrics = {}
        
        # Revenue growth
        if 'Total Revenue' in income_stmt.index:
            revenue_data = income_stmt.loc['Total Revenue']
            if len(revenue_data) >= 2:
                latest = revenue_data.iloc[0]
                previous = revenue_data.iloc[1]
                growth_metrics['revenue_growth_yoy'] = ((latest - previous) / previous * 100)
                
                # Calculate CAGR if we have more years
                if len(revenue_data) >= 3:
                    oldest = revenue_data.iloc[-1]
                    years = len(revenue_data) - 1
                    cagr = ((latest / oldest) ** (1/years) - 1) * 100
                    growth_metrics['revenue_cagr'] = cagr
        
        # Net income growth
        if 'Net Income' in income_stmt.index:
            income_data = income_stmt.loc['Net Income']
            if len(income_data) >= 2:
                latest = income_data.iloc[0]
                previous = income_data.iloc[1]
                if previous != 0:
                    growth_metrics['income_growth_yoy'] = ((latest - previous) / previous * 100)
        
        return growth_metrics
        
    except Exception as e:
        return None
