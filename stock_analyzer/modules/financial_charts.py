"""
Financial Charts Module
Handles creation of all financial statement charts and visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st

def create_combined_income_chart(income_stmt):
    """
    Create combined revenue and net income chart with thinner bars
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        plotly.graph_objects.Figure: Combined income chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
        
        # Get data
        revenue_data = income_stmt.loc['Total Revenue'].dropna() if 'Total Revenue' in income_stmt.index else pd.Series()
        income_data = income_stmt.loc['Net Income'].dropna() if 'Net Income' in income_stmt.index else pd.Series()
        
        if revenue_data.empty and income_data.empty:
            return None
        
        # Align data to common years
        years = list(set(revenue_data.index) | set(income_data.index))
        years = sorted([str(year)[:4] for year in years])
        
        revenue_values = []
        income_values = []
        
        for year in years:
            # Find matching year in data
            rev_val = 0
            inc_val = 0
            
            for date in revenue_data.index:
                if str(date)[:4] == year:
                    rev_val = revenue_data[date] / 1e9
                    break
                    
            for date in income_data.index:
                if str(date)[:4] == year:
                    inc_val = income_data[date] / 1e9
                    break
            
            revenue_values.append(rev_val if rev_val != 0 else None)
            income_values.append(inc_val if inc_val != 0 else None)
        
        # Create the combined chart
        fig = go.Figure()
        
        # Add revenue bars
        fig.add_trace(go.Bar(
            x=years,
            y=revenue_values,
            name='Revenue',
            marker_color='#2E86AB',
            width=0.35,  # Thinner bars
            offsetgroup=1
        ))
        
        # Add net income bars
        fig.add_trace(go.Bar(
            x=years,
            y=income_values,
            name='Net Income',
            marker_color='#2E8B57',
            width=0.35,  # Thinner bars
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Historical Revenue vs Net Income (Billions USD)',
            xaxis_title='Year',
            yaxis_title='Amount (Billions USD)',
            barmode='group',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating cash conversion cycle chart: {str(e)}")
        return None

def create_dupont_analysis_chart(income_stmt, balance_sheet):
    """
    Create DuPont analysis chart showing ROE breakdown
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        
    Returns:
        plotly.graph_objects.Figure: DuPont analysis chart
    """
    try:
        if income_stmt is None or balance_sheet is None:
            return None
            
        years = []
        net_margins = []
        asset_turnovers = []
        equity_multipliers = []
        roe_values = []
        
        for i in range(min(len(income_stmt.columns), len(balance_sheet.columns))):
            year_income = income_stmt.iloc[:, i]
            year_balance = balance_sheet.iloc[:, i]
            year = str(income_stmt.columns[i])[:4]
            
            revenue = year_income.get('Total Revenue', 0)
            net_income = year_income.get('Net Income', 0)
            total_assets = year_balance.get('Total Assets', 0)
            total_equity = year_balance.get('Stockholders Equity', 0) or year_balance.get('Total Stockholder Equity', 0)
            
            if revenue > 0 and total_assets > 0 and total_equity > 0 and net_income != 0:
                net_margin = net_income / revenue
                asset_turnover = revenue / total_assets
                equity_multiplier = total_assets / total_equity
                roe = net_income / total_equity
                
                years.append(year)
                net_margins.append(net_margin * 100)
                asset_turnovers.append(asset_turnover)
                equity_multipliers.append(equity_multiplier)
                roe_values.append(roe * 100)
        
        if not years:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Net Margin (%)', 'Asset Turnover', 'Equity Multiplier', 'ROE (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Net Margin
        fig.add_trace(go.Scatter(x=years, y=net_margins, name='Net Margin %', 
                                line=dict(color='#2E86AB', width=3), mode='lines+markers'), row=1, col=1)
        
        # Asset Turnover
        fig.add_trace(go.Scatter(x=years, y=asset_turnovers, name='Asset Turnover',
                                line=dict(color='#F18F01', width=3), mode='lines+markers'), row=1, col=2)
        
        # Equity Multiplier
        fig.add_trace(go.Scatter(x=years, y=equity_multipliers, name='Equity Multiplier',
                                line=dict(color='#DC143C', width=3), mode='lines+markers'), row=2, col=1)
        
        # ROE
        fig.add_trace(go.Scatter(x=years, y=roe_values, name='ROE %',
                                line=dict(color='#2E8B57', width=3), mode='lines+markers'), row=2, col=2)
        
        fig.update_layout(
            height=600,
            title_text="DuPont Analysis - ROE Breakdown",
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating DuPont analysis chart: {str(e)}")
        return None

def create_growth_rates_chart(income_stmt):
    """
    Create growth rates chart for key metrics
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        plotly.graph_objects.Figure: Growth rates chart
    """
    try:
        if income_stmt is None or income_stmt.empty or len(income_stmt.columns) < 2:
            return None
            
        years = []
        revenue_growth = []
        income_growth = []
        
        for i in range(1, len(income_stmt.columns)):
            current_year = income_stmt.iloc[:, i-1]  # More recent
            previous_year = income_stmt.iloc[:, i]   # Older
            year = str(income_stmt.columns[i-1])[:4]
            
            # Revenue growth
            current_revenue = current_year.get('Total Revenue', 0)
            previous_revenue = previous_year.get('Total Revenue', 0)
            
            # Net income growth
            current_income = current_year.get('Net Income', 0)
            previous_income = previous_year.get('Net Income', 0)
            
            if previous_revenue > 0 and previous_income != 0:
                rev_growth = ((current_revenue - previous_revenue) / previous_revenue * 100)
                inc_growth = ((current_income - previous_income) / previous_income * 100) if previous_income != 0 else 0
                
                years.append(year)
                revenue_growth.append(rev_growth)
                income_growth.append(inc_growth)
        
        if not years:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=revenue_growth,
            name='Revenue Growth %',
            marker_color='#2E86AB',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            x=years,
            y=income_growth,
            name='Net Income Growth %',
            marker_color='#2E8B57',
            opacity=0.8
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Year-over-Year Growth Rates',
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            height=400,
            template='plotly_white',
            barmode='group'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating growth rates chart: {str(e)}")
        return None

def create_debt_analysis_chart(balance_sheet):
    """
    Create debt analysis chart
    
    Args:
        balance_sheet (pd.DataFrame): Balance sheet data
        
    Returns:
        plotly.graph_objects.Figure: Debt analysis chart
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return None
            
        years = []
        total_debt = []
        long_term_debt = []
        short_term_debt = []
        debt_to_assets = []
        
        for i, col in enumerate(balance_sheet.columns):
            year_data = balance_sheet.iloc[:, i]
            year = str(col)[:4]
            
            total_assets = year_data.get('Total Assets', 0)
            tot_debt = year_data.get('Total Debt', 0)
            lt_debt = year_data.get('Long Term Debt', 0)
            st_debt = year_data.get('Current Debt', 0)
            
            if total_assets > 0:
                years.append(year)
                total_debt.append(tot_debt / 1e9 if tot_debt else 0)
                long_term_debt.append(lt_debt / 1e9 if lt_debt else 0)
                short_term_debt.append(st_debt / 1e9 if st_debt else 0)
                debt_to_assets.append((tot_debt / total_assets * 100) if tot_debt else 0)
        
        if not years:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Debt Composition (Billions USD)', 'Debt-to-Assets Ratio (%)'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Debt composition
        fig.add_trace(go.Bar(x=years, y=long_term_debt, name='Long-term Debt', 
                            marker_color='#DC143C'), row=1, col=1)
        fig.add_trace(go.Bar(x=years, y=short_term_debt, name='Short-term Debt',
                            marker_color='#FF6347'), row=1, col=1)
        
        # Debt-to-assets ratio
        fig.add_trace(go.Scatter(x=years, y=debt_to_assets, name='Debt-to-Assets %',
                                line=dict(color='#2E8B57', width=3), mode='lines+markers'), row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="Debt Analysis",
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating debt analysis chart: {str(e)}")
        return None

def create_profitability_tree_chart(income_stmt):
    """
    Create profitability breakdown tree chart
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        plotly.graph_objects.Figure: Profitability tree chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
            
        # Get latest year data
        latest_data = income_stmt.iloc[:, 0]
        year = str(income_stmt.columns[0])[:4]
        
        revenue = latest_data.get('Total Revenue', 0)
        if revenue <= 0:
            return None
            
        # Calculate percentages of revenue
        metrics = {}
        items = [
            'Total Revenue', 'Cost Of Goods Sold', 'Gross Profit',
            'Research And Development', 'Selling General And Administrative',
            'Operating Income', 'Interest Expense', 'Tax Provision', 'Net Income'
        ]
        
        for item in items:
            value = latest_data.get(item, 0)
            if item == 'Total Revenue':
                metrics[item] = 100
            else:
                metrics[item] = (abs(value) / revenue * 100) if value != 0 else 0
        
        # Create waterfall-style breakdown
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Color coding
        colors = ['#2E86AB' if 'Revenue' in label or 'Profit' in label or 'Income' in label 
                 else '#DC143C' for label in labels]
        
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Profitability Breakdown - {year} (% of Revenue)',
            xaxis_title='Financial Metrics',
            yaxis_title='Percentage of Revenue (%)',
            height=500,
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating profitability tree chart: {str(e)}")
        return None
    #Error creating combined income chart: {str(e)}")
     #   return None

def create_combined_balance_sheet_chart(balance_sheet):
    """
    Create combined assets vs liabilities chart
    
    Args:
        balance_sheet (pd.DataFrame): Balance sheet data
        
    Returns:
        plotly.graph_objects.Figure: Combined balance sheet chart
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return None
        
        # Get data
        assets_data = balance_sheet.loc['Total Assets'].dropna() if 'Total Assets' in balance_sheet.index else pd.Series()
        
        # Find liabilities - try different naming conventions
        liabilities_data = pd.Series()
        liab_names = ['Total Liabilities Net Minority Interest', 'Total Liab', 'Total Liabilities']
        for name in liab_names:
            if name in balance_sheet.index:
                liabilities_data = balance_sheet.loc[name].dropna()
                break
        
        # If no direct liabilities, calculate as Total Assets - Shareholders Equity
        if liabilities_data.empty:
            equity_names = ['Stockholders Equity', 'Total Stockholder Equity']
            for name in equity_names:
                if name in balance_sheet.index:
                    equity_data = balance_sheet.loc[name].dropna()
                    if not equity_data.empty and not assets_data.empty:
                        # Align the dates
                        common_dates = assets_data.index.intersection(equity_data.index)
                        liabilities_data = assets_data[common_dates] - equity_data[common_dates]
                    break
        
        if assets_data.empty:
            return None
        
        # Align data to common years
        years = list(set(assets_data.index) | set(liabilities_data.index))
        years = sorted([str(year)[:4] for year in years])
        
        assets_values = []
        liabilities_values = []
        
        for year in years:
            # Find matching year in data
            asset_val = 0
            liab_val = 0
            
            for date in assets_data.index:
                if str(date)[:4] == year:
                    asset_val = assets_data[date] / 1e9
                    break
                    
            for date in liabilities_data.index:
                if str(date)[:4] == year:
                    liab_val = liabilities_data[date] / 1e9
                    break
            
            assets_values.append(asset_val if asset_val != 0 else None)
            liabilities_values.append(liab_val if liab_val != 0 else None)
        
        # Create the combined chart
        fig = go.Figure()
        
        # Add assets bars
        fig.add_trace(go.Bar(
            x=years,
            y=assets_values,
            name='Total Assets',
            marker_color='#2E86AB',
            width=0.35,  # Thinner bars
            offsetgroup=1
        ))
        
        # Add liabilities bars
        if not all(v is None for v in liabilities_values):
            fig.add_trace(go.Bar(
                x=years,
                y=liabilities_values,
                name='Total Liabilities',
                marker_color='#DC143C',
                width=0.35,  # Thinner bars
                offsetgroup=2
            ))
        
        fig.update_layout(
            title='Historical Assets vs Liabilities (Billions USD)',
            xaxis_title='Year',
            yaxis_title='Amount (Billions USD)',
            barmode='group',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating combined balance sheet chart: {str(e)}")
        return None

def create_historical_performance_chart(income_stmt, balance_sheet, cash_flow):
    """
    Create historical performance bar charts for all statements
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        cash_flow (pd.DataFrame): Cash flow data
        
    Returns:
        dict: Dictionary of chart figures
    """
    charts = {}
    
    # Income Statement Charts
    if income_stmt is not None and not income_stmt.empty:
        # Revenue chart
        if 'Total Revenue' in income_stmt.index:
            revenue_data = income_stmt.loc['Total Revenue'].dropna()
            charts['revenue'] = px.bar(
                x=[str(col)[:4] for col in revenue_data.index],
                y=revenue_data.values / 1e9,
                title='Historical Revenue (Billions USD)',
                labels={'x': 'Year', 'y': 'Revenue (Billions USD)'},
                color_discrete_sequence=['#2E86AB']
            )
            charts['revenue'].update_layout(height=400)
        
        # Net Income chart
        if 'Net Income' in income_stmt.index:
            income_data = income_stmt.loc['Net Income'].dropna()
            charts['net_income'] = px.bar(
                x=[str(col)[:4] for col in income_data.index],
                y=income_data.values / 1e9,
                title='Historical Net Income (Billions USD)',
                labels={'x': 'Year', 'y': 'Net Income (Billions USD)'},
                color_discrete_sequence=['#2E8B57']
            )
            charts['net_income'].update_layout(height=400)
    
    # Balance Sheet Charts
    if balance_sheet is not None and not balance_sheet.empty:
        # Total Assets
        if 'Total Assets' in balance_sheet.index:
            assets_data = balance_sheet.loc['Total Assets'].dropna()
            charts['total_assets'] = px.bar(
                x=[str(col)[:4] for col in assets_data.index],
                y=assets_data.values / 1e9,
                title='Historical Total Assets (Billions USD)',
                labels={'x': 'Year', 'y': 'Total Assets (Billions USD)'},
                color_discrete_sequence=['#F18F01']
            )
            charts['total_assets'].update_layout(height=400)
        
        # Total Debt
        debt_metrics = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
        for debt_metric in debt_metrics:
            if debt_metric in balance_sheet.index:
                debt_data = balance_sheet.loc[debt_metric].dropna()
                charts['total_debt'] = px.bar(
                    x=[str(col)[:4] for col in debt_data.index],
                    y=debt_data.values / 1e9,
                    title='Historical Total Debt (Billions USD)',
                    labels={'x': 'Year', 'y': 'Total Debt (Billions USD)'},
                    color_discrete_sequence=['#DC143C']
                )
                charts['total_debt'].update_layout(height=400)
                break
        
        # Shareholders Equity
        equity_metrics = ['Stockholders Equity', 'Total Stockholder Equity']
        for equity_metric in equity_metrics:
            if equity_metric in balance_sheet.index:
                equity_data = balance_sheet.loc[equity_metric].dropna()
                charts['equity'] = px.bar(
                    x=[str(col)[:4] for col in equity_data.index],
                    y=equity_data.values / 1e9,
                    title='Historical Shareholders Equity (Billions USD)',
                    labels={'x': 'Year', 'y': 'Equity (Billions USD)'},
                    color_discrete_sequence=['#8A2BE2']
                )
                charts['equity'].update_layout(height=400)
                break
    
    # Cash Flow Charts
    if cash_flow is not None and not cash_flow.empty:
        # Operating Cash Flow
        ocf_metrics = ['Operating Cash Flow', 'Total Cash From Operating Activities']
        for ocf_metric in ocf_metrics:
            if ocf_metric in cash_flow.index:
                ocf_data = cash_flow.loc[ocf_metric].dropna()
                charts['operating_cf'] = px.bar(
                    x=[str(col)[:4] for col in ocf_data.index],
                    y=ocf_data.values / 1e9,
                    title='Historical Operating Cash Flow (Billions USD)',
                    labels={'x': 'Year', 'y': 'Operating CF (Billions USD)'},
                    color_discrete_sequence=['#32CD32']
                )
                charts['operating_cf'].update_layout(height=400)
                break
        
        # Free Cash Flow (Operating CF - Capital Expenditures)
        if ('Operating Cash Flow' in cash_flow.index or 'Total Cash From Operating Activities' in cash_flow.index) and 'Capital Expenditures' in cash_flow.index:
            ocf_key = 'Operating Cash Flow' if 'Operating Cash Flow' in cash_flow.index else 'Total Cash From Operating Activities'
            ocf_data = cash_flow.loc[ocf_key].dropna()
            capex_data = cash_flow.loc['Capital Expenditures'].dropna()
            
            # Align the data
            common_dates = ocf_data.index.intersection(capex_data.index)
            if len(common_dates) > 0:
                fcf_data = ocf_data[common_dates] + capex_data[common_dates]  # CapEx is negative
                charts['free_cf'] = px.bar(
                    x=[str(col)[:4] for col in fcf_data.index],
                    y=fcf_data.values / 1e9,
                    title='Historical Free Cash Flow (Billions USD)',
                    labels={'x': 'Year', 'y': 'Free CF (Billions USD)'},
                    color_discrete_sequence=['#20B2AA']
                )
                charts['free_cf'].update_layout(height=400)
    
    return charts

def create_historical_ratios_chart(income_stmt, balance_sheet):
    """
    Create historical ratios chart
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        
    Returns:
        plotly.graph_objects.Figure: Historical ratios chart
    """
    try:
        if income_stmt is None or balance_sheet is None:
            return None
            
        years = []
        gross_margins = []
        net_margins = []
        roa_values = []
        roe_values = []
        current_ratios = []
        debt_equity_ratios = []
        
        # Calculate ratios for each available year
        for i in range(min(len(income_stmt.columns), len(balance_sheet.columns))):
            year_income = income_stmt.iloc[:, i]
            year_balance = balance_sheet.iloc[:, i]
            year = str(income_stmt.columns[i])[:4]
            
            revenue = year_income.get('Total Revenue', 0)
            net_income = year_income.get('Net Income', 0)
            gross_profit = year_income.get('Gross Profit', 0)
            total_assets = year_balance.get('Total Assets', 0)
            total_equity = year_balance.get('Stockholders Equity', 0) or year_balance.get('Total Stockholder Equity', 0)
            current_assets = year_balance.get('Current Assets', 0)
            current_liabilities = year_balance.get('Current Liabilities', 0)
            total_debt = year_balance.get('Total Debt', 0)
            
            if revenue > 0 and net_income != 0 and total_assets > 0:
                years.append(year)
                gross_margins.append((gross_profit / revenue * 100) if gross_profit else 0)
                net_margins.append(net_income / revenue * 100)
                roa_values.append(net_income / total_assets * 100)
                roe_values.append((net_income / total_equity * 100) if total_equity > 0 else 0)
                current_ratios.append((current_assets / current_liabilities) if current_liabilities > 0 else 0)
                debt_equity_ratios.append((total_debt / total_equity) if total_equity > 0 else 0)
        
        if not years:
            return None
        
        # Create subplots for different ratio categories
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profitability Ratios (%)', 'Return Ratios (%)', 'Liquidity Ratio', 'Leverage Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Profitability ratios
        fig.add_trace(
            go.Scatter(x=years, y=gross_margins, name='Gross Margin %', line=dict(color='#2E86AB')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=years, y=net_margins, name='Net Margin %', line=dict(color='#2E8B57')),
            row=1, col=1
        )
        
        # Return ratios
        fig.add_trace(
            go.Scatter(x=years, y=roa_values, name='ROA %', line=dict(color='#F18F01')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=years, y=roe_values, name='ROE %', line=dict(color='#DC143C')),
            row=1, col=2
        )
        
        # Liquidity ratio
        fig.add_trace(
            go.Scatter(x=years, y=current_ratios, name='Current Ratio', line=dict(color='#8A2BE2')),
            row=2, col=1
        )
        
        # Leverage ratio
        fig.add_trace(
            go.Scatter(x=years, y=debt_equity_ratios, name='Debt/Equity', line=dict(color='#FF6347')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Historical Financial Ratios",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating historical ratios chart: {str(e)}")
        return None

def create_waterfall_chart(income_stmt, year):
    """
    Create proper waterfall chart showing revenue breakdown to net income
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        year (str): Selected year for analysis
        
    Returns:
        plotly.graph_objects.Figure: Waterfall chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
            
        # Get the selected year data
        year_col = None
        for col in income_stmt.columns:
            if str(year) in str(col):
                year_col = col
                break
        
        if year_col is None:
            return None
        
        # Define the waterfall components in logical order
        waterfall_items = [
            ('Total Revenue', 'relative', '#2E86AB'),
            ('Cost Of Goods Sold', 'relative', '#A23B72'),
            ('Gross Profit', 'total', '#F18F01'),
            ('Research And Development', 'relative', '#C73E1D'),
            ('Selling General And Administrative', 'relative', '#C73E1D'),
            ('Operating Income', 'total', '#F18F01'),
            ('Interest Expense', 'relative', '#A23B72'),
            ('Tax Provision', 'relative', '#A23B72'),
            ('Net Income', 'total', '#2E8B57')
        ]
        
        # Extract values and prepare data
        x_labels = []
        y_values = []
        measures = []
        text_values = []
        
        for item, measure, color in waterfall_items:
            if item in income_stmt.index:
                value = income_stmt.loc[item, year_col]
                if pd.notna(value):
                    x_labels.append(item.replace(' And ', ' & '))
                    
                    # Convert negative expenses to positive for waterfall display
                    if item in ['Cost Of Goods Sold', 'Research And Development', 
                               'Selling General And Administrative', 'Interest Expense', 'Tax Provision']:
                        y_values.append(-abs(value))  # Make expenses negative
                        text_values.append(f"-${abs(value)/1e9:.2f}B")
                    else:
                        y_values.append(value)
                        text_values.append(f"${value/1e9:.2f}B")
                    
                    measures.append(measure)
        
        if not x_labels:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Waterfall(
            name="Income Statement Flow",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=text_values,
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2E8B57"}},
            decreasing={"marker": {"color": "#DC143C"}},
            totals={"marker": {"color": "#F18F01"}}
        ))
        
        fig.update_layout(
            title=f"Income Statement Waterfall - {year}",
            showlegend=False,
            height=500,
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return None

def create_sankey_diagram(income_stmt, year):
    """
    Create Sankey diagram for income statement flow
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        year (str): Selected year for analysis
        
    Returns:
        plotly.graph_objects.Figure: Sankey diagram
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
            
        # Get the selected year data
        year_col = None
        for col in income_stmt.columns:
            if str(year) in str(col):
                year_col = col
                break
        
        if year_col is None:
            return None
        
        # Get values (in billions for better readability)
        revenue = income_stmt.loc['Total Revenue', year_col] / 1e9 if 'Total Revenue' in income_stmt.index else 0
        cogs = abs(income_stmt.loc['Cost Of Goods Sold', year_col]) / 1e9 if 'Cost Of Goods Sold' in income_stmt.index else 0
        rd = abs(income_stmt.loc['Research And Development', year_col]) / 1e9 if 'Research And Development' in income_stmt.index else 0
        sga = abs(income_stmt.loc['Selling General And Administrative', year_col]) / 1e9 if 'Selling General And Administrative' in income_stmt.index else 0
        interest = abs(income_stmt.loc['Interest Expense', year_col]) / 1e9 if 'Interest Expense' in income_stmt.index else 0
        tax = abs(income_stmt.loc['Tax Provision', year_col]) / 1e9 if 'Tax Provision' in income_stmt.index else 0
        net_income = income_stmt.loc['Net Income', year_col] / 1e9 if 'Net Income' in income_stmt.index else 0
        
        # Create labels and flows
        labels = ['Total Revenue', 'COGS', 'R&D', 'SG&A', 'Interest', 'Tax', 'Net Income']
        sources = []
        targets = []
        values = []
        
        # Define flows from revenue to expenses and net income
        if revenue > 0:
            if cogs > 0:
                sources.append(0)  # Revenue
                targets.append(1)  # COGS
                values.append(cogs)
            
            if rd > 0:
                sources.append(0)  # Revenue
                targets.append(2)  # R&D
                values.append(rd)
            
            if sga > 0:
                sources.append(0)  # Revenue
                targets.append(3)  # SG&A
                values.append(sga)
            
            if interest > 0:
                sources.append(0)  # Revenue
                targets.append(4)  # Interest
                values.append(interest)
            
            if tax > 0:
                sources.append(0)  # Revenue
                targets.append(5)  # Tax
                values.append(tax)
            
            if net_income > 0:
                sources.append(0)  # Revenue
                targets.append(6)  # Net Income
                values.append(net_income)
        
        if not sources:
            return None
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#2E86AB", "#DC143C", "#DC143C", "#DC143C", "#DC143C", "#DC143C", "#2E8B57"]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=["rgba(255,0,0,0.3)" if i != len(sources)-1 else "rgba(0,128,0,0.3)" for i in range(len(sources))]
            )
        )])
        
        fig.update_layout(
            title_text=f"Income Statement Flow - {year} (Billions USD)",
            font_size=12,
            height=500,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Sankey diagram: {str(e)}")
        return None

def create_margin_analysis_chart(income_stmt):
    """
    Create margin analysis over time
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        plotly.graph_objects.Figure: Margin analysis chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
        
        years = []
        gross_margins = []
        operating_margins = []
        net_margins = []
        
        for i, col in enumerate(income_stmt.columns):
            year_data = income_stmt.iloc[:, i]
            year = str(col)[:4]
            
            revenue = year_data.get('Total Revenue', 0)
            if revenue <= 0:
                continue
                
            gross_profit = year_data.get('Gross Profit', 0)
            operating_income = year_data.get('Operating Income', 0)
            net_income = year_data.get('Net Income', 0)
            
            years.append(year)
            gross_margins.append((gross_profit / revenue * 100) if gross_profit else 0)
            operating_margins.append((operating_income / revenue * 100) if operating_income else 0)
            net_margins.append(net_income / revenue * 100)
        
        if not years:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=gross_margins, name='Gross Margin %',
            line=dict(color='#2E86AB', width=3), mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=years, y=operating_margins, name='Operating Margin %',
            line=dict(color='#F18F01', width=3), mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=years, y=net_margins, name='Net Margin %',
            line=dict(color='#2E8B57', width=3), mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Margin Analysis Over Time',
            xaxis_title='Year',
            yaxis_title='Margin (%)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating margin analysis chart: {str(e)}")
        return None

def create_cash_conversion_cycle_chart(balance_sheet, income_stmt):
    """
    Create cash conversion cycle analysis
    
    Args:
        balance_sheet (pd.DataFrame): Balance sheet data
        income_stmt (pd.DataFrame): Income statement data
        
    Returns:
        plotly.graph_objects.Figure: Cash conversion cycle chart
    """
    try:
        if balance_sheet is None or income_stmt is None:
            return None
            
        years = []
        dso_values = []  # Days Sales Outstanding
        dio_values = []  # Days Inventory Outstanding
        dpo_values = []  # Days Payable Outstanding
        ccc_values = []  # Cash Conversion Cycle
        
        for i in range(min(len(balance_sheet.columns), len(income_stmt.columns))):
            bs_data = balance_sheet.iloc[:, i]
            is_data = income_stmt.iloc[:, i]
            year = str(balance_sheet.columns[i])[:4]
            
            # Get required data
            revenue = is_data.get('Total Revenue', 0)
            cogs = is_data.get('Cost Of Goods Sold', 0)
            accounts_receivable = bs_data.get('Accounts Receivable', 0)
            inventory = bs_data.get('Inventory', 0)
            accounts_payable = bs_data.get('Accounts Payable', 0)
            
            if revenue <= 0 or cogs <= 0:
                continue
                
            # Calculate ratios
            dso = (accounts_receivable / revenue * 365) if accounts_receivable else 0
            dio = (inventory / cogs * 365) if inventory else 0
            dpo = (accounts_payable / cogs * 365) if accounts_payable else 0
            ccc = dso + dio - dpo
            
            years.append(year)
            dso_values.append(dso)
            dio_values.append(dio)
            dpo_values.append(dpo)
            ccc_values.append(ccc)
        
        if not years:
            return None
            
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Working Capital Components', 'Cash Conversion Cycle'))
        
        # Components
        fig.add_trace(go.Bar(x=years, y=dso_values, name='DSO', marker_color='#2E86AB'), row=1, col=1)
        fig.add_trace(go.Bar(x=years, y=dio_values, name='DIO', marker_color='#F18F01'), row=1, col=1)
        fig.add_trace(go.Bar(x=years, y=dpo_values, name='DPO', marker_color='#DC143C'), row=1, col=1)
        
        # Cash Conversion Cycle
        fig.add_trace(go.Scatter(x=years, y=ccc_values, name='CCC', line=dict(color='#2E8B57', width=3),
                                mode='lines+markers'), row=2, col=1)
        
        fig.update_layout(height=600, template='plotly_white', title='Working Capital Management')
        
        return fig
        
    except Exception as e:
        st.error(f"")