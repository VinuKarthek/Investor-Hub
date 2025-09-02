"""
Optimized Financial Charts Module
Handles creation of all financial statement charts and visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from typing import Optional, Dict, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for better maintainability
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#2E8B57', 
    'accent': '#F18F01',
    'danger': '#DC143C',
    'success': '#32CD32',
    'warning': '#FF6347',
    'info': '#8A2BE2',
    'light': '#20B2AA'
}

FINANCIAL_METRICS = {
    'revenue': ['Total Revenue', 'Revenue'],
    'net_income': ['Net Income', 'Net Income Common Stockholders'],
    'gross_profit': ['Gross Profit'],
    'operating_income': ['Operating Income', 'Operating Revenue'],
    'total_assets': ['Total Assets'],
    'total_liabilities': ['Total Liabilities Net Minority Interest', 'Total Liab', 'Total Liabilities'],
    'stockholders_equity': ['Stockholders Equity', 'Total Stockholder Equity', 'Shareholders Equity'],
    'total_debt': ['Total Debt', 'Long Term Debt', 'Short Long Term Debt'],
    'operating_cf': ['Operating Cash Flow', 'Total Cash From Operating Activities'],
    'capex': ['Capital Expenditures', 'Capital Expenditure'],
    'current_assets': ['Current Assets'],
    'current_liabilities': ['Current Liabilities'],
    'accounts_receivable': ['Accounts Receivable', 'Net Receivables'],
    'inventory': ['Inventory'],
    'accounts_payable': ['Accounts Payable']
}

class FinancialChartsOptimized:
    """Optimized financial charts class with caching and error handling"""
    
    def __init__(self):
        self.cache = {}
    
    @staticmethod
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def _safe_get_metric(df: pd.DataFrame, metric_names: List[str], default: float = 0) -> pd.Series:
        """Safely get financial metric from dataframe with fallback options"""
        if df is None or df.empty:
            return pd.Series(dtype=float)
        
        for name in metric_names:
            if name in df.index:
                return df.loc[name].dropna()
        return pd.Series(dtype=float)
    
    @staticmethod
    def _prepare_yearly_data(data: pd.Series) -> Tuple[List[str], List[float]]:
        """Convert financial data to yearly format"""
        if data.empty:
            return [], []
        
        years = [str(date)[:4] for date in data.index]
        values = [val / 1e9 if pd.notna(val) and val != 0 else None for val in data.values]
        return years, values
    
    @staticmethod
    def _create_base_layout(title: str, height: int = 400, xaxis_title: str = 'Year', 
                           yaxis_title: str = 'Amount (Billions USD)') -> Dict:
        """Create base layout configuration for charts"""
        return {
            'title': title,
            'xaxis_title': xaxis_title,
            'yaxis_title': yaxis_title,
            'height': height,
            'template': 'plotly_white',
            'hovermode': 'x unified',
            'margin': dict(l=50, r=50, t=50, b=50)
        }

@st.cache_data(ttl=600)
def create_combined_income_chart(income_stmt: pd.DataFrame) -> Optional[go.Figure]:
    """
    Optimized combined revenue and net income chart
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get data using optimized method
        revenue_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
        income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
        
        if revenue_data.empty and income_data.empty:
            return None
        
        # Prepare data
        rev_years, rev_values = charts._prepare_yearly_data(revenue_data)
        inc_years, inc_values = charts._prepare_yearly_data(income_data)
        
        # Align data to common years
        all_years = sorted(list(set(rev_years + inc_years)))
        
        aligned_revenue = []
        aligned_income = []
        
        for year in all_years:
            rev_val = rev_values[rev_years.index(year)] if year in rev_years else None
            inc_val = inc_values[inc_years.index(year)] if year in inc_years else None
            aligned_revenue.append(rev_val)
            aligned_income.append(inc_val)
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=all_years, y=aligned_revenue,
            name='Revenue', marker_color=COLORS['primary'],
            width=0.35, offsetgroup=1,
            hovertemplate='<b>Revenue</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=all_years, y=aligned_income,
            name='Net Income', marker_color=COLORS['secondary'],
            width=0.35, offsetgroup=2,
            hovertemplate='<b>Net Income</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
        ))
        
        layout = FinancialChartsOptimized._create_base_layout(
            'Historical Revenue vs Net Income (Billions USD)'
        )
        layout['barmode'] = 'group'
        layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        
        fig.update_layout(**layout)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating combined income chart: {str(e)}")
        st.error(f"Error creating income chart: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_enhanced_dupont_analysis(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[go.Figure]:
    """
    Enhanced DuPont analysis with better error handling and performance
    """
    try:
        if income_stmt is None or balance_sheet is None:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get required metrics
        revenue_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
        income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
        assets_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_assets'])
        equity_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
        
        # Calculate ratios
        ratios_data = []
        for date in income_data.index:
            if date in revenue_data.index and date in assets_data.index and date in equity_data.index:
                year = str(date)[:4]
                revenue = revenue_data[date]
                net_income = income_data[date]
                total_assets = assets_data[date]
                total_equity = equity_data[date]
                
                if all(val > 0 for val in [revenue, total_assets, total_equity]) and net_income != 0:
                    ratios_data.append({
                        'year': year,
                        'net_margin': (net_income / revenue) * 100,
                        'asset_turnover': revenue / total_assets,
                        'equity_multiplier': total_assets / total_equity,
                        'roe': (net_income / total_equity) * 100
                    })
        
        if not ratios_data:
            return None
        
        df_ratios = pd.DataFrame(ratios_data)
        
        # Create enhanced subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Net Margin (%)', 'Asset Turnover', 'Equity Multiplier', 'ROE (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12
        )
        
        # Add traces with enhanced styling
        traces = [
            (df_ratios['net_margin'], 'Net Margin %', COLORS['primary'], 1, 1),
            (df_ratios['asset_turnover'], 'Asset Turnover', COLORS['accent'], 1, 2),
            (df_ratios['equity_multiplier'], 'Equity Multiplier', COLORS['danger'], 2, 1),
            (df_ratios['roe'], 'ROE %', COLORS['secondary'], 2, 2)
        ]
        
        for values, name, color, row, col in traces:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=values, name=name,
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name}</b><br>Year: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ), row=row, col=col
            )
        
        fig.update_layout(
            height=600,
            title_text="Enhanced DuPont Analysis - ROE Breakdown",
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating DuPont analysis: {str(e)}")
        st.error(f"Error creating DuPont analysis: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_comprehensive_growth_analysis(income_stmt: pd.DataFrame) -> Optional[go.Figure]:
    """
    Comprehensive growth analysis with multiple metrics
    """
    try:
        if income_stmt is None or income_stmt.empty or len(income_stmt.columns) < 2:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get metrics
        revenue_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
        income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
        gross_profit_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['gross_profit'])
        operating_income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['operating_income'])
        
        growth_data = []
        
        # Calculate year-over-year growth rates
        dates = sorted(revenue_data.index, reverse=True)  # Most recent first
        
        for i in range(len(dates) - 1):
            current_date = dates[i]
            previous_date = dates[i + 1]
            year = str(current_date)[:4]
            
            metrics = {
                'revenue': (revenue_data.get(current_date, 0), revenue_data.get(previous_date, 0)),
                'net_income': (income_data.get(current_date, 0), income_data.get(previous_date, 0)),
                'gross_profit': (gross_profit_data.get(current_date, 0), gross_profit_data.get(previous_date, 0)),
                'operating_income': (operating_income_data.get(current_date, 0), operating_income_data.get(previous_date, 0))
            }
            
            year_growth = {'year': year}
            
            for metric_name, (current, previous) in metrics.items():
                if previous != 0 and pd.notna(current) and pd.notna(previous):
                    growth_rate = ((current - previous) / abs(previous)) * 100
                    year_growth[f'{metric_name}_growth'] = growth_rate
                else:
                    year_growth[f'{metric_name}_growth'] = 0
            
            growth_data.append(year_growth)
        
        if not growth_data:
            return None
        
        df_growth = pd.DataFrame(growth_data)
        
        # Create enhanced growth chart
        fig = go.Figure()
        
        growth_metrics = [
            ('revenue_growth', 'Revenue Growth', COLORS['primary']),
            ('net_income_growth', 'Net Income Growth', COLORS['secondary']),
            ('gross_profit_growth', 'Gross Profit Growth', COLORS['accent']),
            ('operating_income_growth', 'Operating Income Growth', COLORS['info'])
        ]
        
        for metric, name, color in growth_metrics:
            if metric in df_growth.columns:
                fig.add_trace(go.Bar(
                    x=df_growth['year'],
                    y=df_growth[metric],
                    name=name,
                    marker_color=color,
                    opacity=0.8,
                    hovertemplate=f'<b>{name}</b><br>Year: %{{x}}<br>Growth: %{{y:.1f}}%<extra></extra>'
                ))
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        layout = FinancialChartsOptimized._create_base_layout(
            'Comprehensive Year-over-Year Growth Analysis',
            yaxis_title='Growth Rate (%)'
        )
        layout['barmode'] = 'group'
        layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        
        fig.update_layout(**layout)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating growth analysis: {str(e)}")
        st.error(f"Error creating growth analysis: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_advanced_profitability_analysis(income_stmt: pd.DataFrame) -> Optional[go.Figure]:
    """
    Advanced profitability analysis with multiple margin types
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get required data
        revenue_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
        gross_profit_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['gross_profit'])
        operating_income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['operating_income'])
        net_income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
        
        margin_data = []
        
        for date in revenue_data.index:
            year = str(date)[:4]
            revenue = revenue_data[date]
            
            if revenue > 0:
                margins = {'year': year}
                
                # Calculate different margin types
                if date in gross_profit_data.index:
                    margins['gross_margin'] = (gross_profit_data[date] / revenue) * 100
                
                if date in operating_income_data.index:
                    margins['operating_margin'] = (operating_income_data[date] / revenue) * 100
                
                if date in net_income_data.index:
                    margins['net_margin'] = (net_income_data[date] / revenue) * 100
                
                margin_data.append(margins)
        
        if not margin_data:
            return None
        
        df_margins = pd.DataFrame(margin_data)
        
        # Create margin analysis chart
        fig = go.Figure()
        
        margin_types = [
            ('gross_margin', 'Gross Margin', COLORS['primary']),
            ('operating_margin', 'Operating Margin', COLORS['accent']),
            ('net_margin', 'Net Margin', COLORS['secondary'])
        ]
        
        for margin_col, name, color in margin_types:
            if margin_col in df_margins.columns:
                fig.add_trace(go.Scatter(
                    x=df_margins['year'],
                    y=df_margins[margin_col],
                    name=name,
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name}</b><br>Year: %{{x}}<br>Margin: %{{y:.1f}}%<extra></extra>'
                ))
        
        layout = FinancialChartsOptimized._create_base_layout(
            'Advanced Profitability Margin Analysis',
            yaxis_title='Margin (%)'
        )
        layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        
        fig.update_layout(**layout)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating profitability analysis: {str(e)}")
        st.error(f"Error creating profitability analysis: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_financial_health_dashboard(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
                                    cash_flow: pd.DataFrame) -> Optional[go.Figure]:
    """
    Comprehensive financial health dashboard
    """
    try:
        if not any([income_stmt is not None, balance_sheet is not None, cash_flow is not None]):
            return None
        
        charts = FinancialChartsOptimized()
        
        # Calculate key health metrics
        health_data = []
        
        # Get common dates
        dates = set()
        if income_stmt is not None:
            dates.update(income_stmt.columns)
        if balance_sheet is not None:
            dates.update(balance_sheet.columns)
        if cash_flow is not None:
            dates.update(cash_flow.columns)
        
        dates = sorted(dates, reverse=True)
        
        for date in dates[:5]:  # Last 5 years
            year = str(date)[:4]
            health_metrics = {'year': year}
            
            # Profitability metrics
            if income_stmt is not None and date in income_stmt.columns:
                revenue = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
                net_income = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
                
                if not revenue.empty and not net_income.empty and date in revenue.index and date in net_income.index:
                    if revenue[date] > 0:
                        health_metrics['net_margin'] = (net_income[date] / revenue[date]) * 100
            
            # Liquidity metrics
            if balance_sheet is not None and date in balance_sheet.columns:
                current_assets = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['current_assets'])
                current_liabilities = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['current_liabilities'])
                
                if not current_assets.empty and not current_liabilities.empty:
                    if date in current_assets.index and date in current_liabilities.index:
                        if current_liabilities[date] > 0:
                            health_metrics['current_ratio'] = current_assets[date] / current_liabilities[date]
                
                # Leverage metrics
                total_debt = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_debt'])
                total_equity = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
                
                if not total_debt.empty and not total_equity.empty:
                    if date in total_debt.index and date in total_equity.index:
                        if total_equity[date] > 0:
                            health_metrics['debt_to_equity'] = total_debt[date] / total_equity[date]
            
            # Cash flow metrics
            if cash_flow is not None and date in cash_flow.columns:
                operating_cf = charts._safe_get_metric(cash_flow, FINANCIAL_METRICS['operating_cf'])
                if not operating_cf.empty and date in operating_cf.index:
                    health_metrics['operating_cf_billions'] = operating_cf[date] / 1e9
            
            if health_metrics:
                health_data.append(health_metrics)
        
        if not health_data:
            return None
        
        df_health = pd.DataFrame(health_data)
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profitability (Net Margin %)', 'Liquidity (Current Ratio)', 
                          'Leverage (Debt/Equity)', 'Cash Generation (Operating CF $B)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15
        )
        
        # Add health metrics
        if 'net_margin' in df_health.columns:
            fig.add_trace(
                go.Scatter(x=df_health['year'], y=df_health['net_margin'], 
                          name='Net Margin', line=dict(color=COLORS['secondary'], width=3),
                          mode='lines+markers', marker=dict(size=8)), row=1, col=1
            )
        
        if 'current_ratio' in df_health.columns:
            fig.add_trace(
                go.Scatter(x=df_health['year'], y=df_health['current_ratio'],
                          name='Current Ratio', line=dict(color=COLORS['primary'], width=3),
                          mode='lines+markers', marker=dict(size=8)), row=1, col=2
            )
            # Add reference line for healthy current ratio
            fig.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2)
        
        if 'debt_to_equity' in df_health.columns:
            fig.add_trace(
                go.Scatter(x=df_health['year'], y=df_health['debt_to_equity'],
                          name='Debt/Equity', line=dict(color=COLORS['danger'], width=3),
                          mode='lines+markers', marker=dict(size=8)), row=2, col=1
            )
        
        if 'operating_cf_billions' in df_health.columns:
            fig.add_trace(
                go.Bar(x=df_health['year'], y=df_health['operating_cf_billions'],
                      name='Operating CF', marker_color=COLORS['success']), row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            title_text="Financial Health Dashboard",
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating financial health dashboard: {str(e)}")
        st.error(f"Error creating financial health dashboard: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_optimized_waterfall_chart(income_stmt: pd.DataFrame, year: str) -> Optional[go.Figure]:
    """
    Optimized waterfall chart with better data handling
    """
    try:
        if income_stmt is None or income_stmt.empty:
            return None
        
        # Find the correct year column
        year_col = None
        for col in income_stmt.columns:
            if str(year) in str(col):
                year_col = col
                break
        
        if year_col is None:
            return None
        
        year_data = income_stmt[year_col]
        
        # Define waterfall structure with improved logic
        waterfall_structure = [
            ('Total Revenue', 'absolute', COLORS['primary'], 'positive'),
            ('Cost Of Goods Sold', 'relative', COLORS['danger'], 'negative'),
            ('Gross Profit', 'total', COLORS['accent'], 'total'),
            ('Research And Development', 'relative', COLORS['warning'], 'negative'),
            ('Selling General And Administrative', 'relative', COLORS['warning'], 'negative'),
            ('Operating Income', 'total', COLORS['accent'], 'total'),
            ('Interest Expense', 'relative', COLORS['danger'], 'negative'),
            ('Tax Provision', 'relative', COLORS['danger'], 'negative'),
            ('Net Income', 'total', COLORS['secondary'], 'total')
        ]
        
        # Prepare waterfall data
        x_labels = []
        y_values = []
        measures = []
        text_values = []
        colors = []
        
        for item, measure, color, value_type in waterfall_structure:
            if item in year_data.index and pd.notna(year_data[item]):
                value = year_data[item]
                
                x_labels.append(item.replace(' And ', ' & '))
                measures.append(measure)
                
                # Handle different value types
                if value_type == 'negative' and value > 0:
                    # Convert positive expenses to negative for waterfall
                    display_value = -value
                    text_values.append(f"-${value/1e9:.2f}B")
                elif value_type == 'negative' and value < 0:
                    # Already negative
                    display_value = value
                    text_values.append(f"${abs(value)/1e9:.2f}B")
                else:
                    # Positive values (revenue, profits)
                    display_value = value
                    text_values.append(f"${value/1e9:.2f}B")
                
                y_values.append(display_value)
        
        if not x_labels:
            return None
        
        # Create enhanced waterfall chart
        fig = go.Figure()
        
        fig.add_trace(go.Waterfall(
            name="Income Statement Flow",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=text_values,
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
            increasing={"marker": {"color": COLORS['secondary']}},
            decreasing={"marker": {"color": COLORS['danger']}},
            totals={"marker": {"color": COLORS['accent']}},
            hovertemplate='<b>%{x}</b><br>Amount: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Income Statement Waterfall Analysis - {year}",
            showlegend=False,
            height=600,
            xaxis_tickangle=-45,
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=100)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating waterfall chart: {str(e)}")
        st.error(f"Error creating waterfall chart: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_comprehensive_balance_sheet_analysis(balance_sheet: pd.DataFrame) -> Optional[go.Figure]:
    """
    Comprehensive balance sheet analysis with multiple views
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get key balance sheet metrics
        assets_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_assets'])
        liabilities_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_liabilities'])
        equity_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
        debt_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_debt'])
        
        # If no direct liabilities, calculate from assets - equity
        if liabilities_data.empty and not assets_data.empty and not equity_data.empty:
            common_dates = assets_data.index.intersection(equity_data.index)
            liabilities_data = assets_data[common_dates] - equity_data[common_dates]
        
        # Prepare data
        years = []
        assets_values = []
        liabilities_values = []
        equity_values = []
        debt_values = []
        
        for date in assets_data.index:
            year = str(date)[:4]
            years.append(year)
            assets_values.append(assets_data[date] / 1e9)
            
            liab_val = liabilities_data[date] / 1e9 if date in liabilities_data.index else 0
            liabilities_values.append(liab_val)
            
            equity_val = equity_data[date] / 1e9 if date in equity_data.index else 0
            equity_values.append(equity_val)
            
            debt_val = debt_data[date] / 1e9 if date in debt_data.index else 0
            debt_values.append(debt_val)
        
        if not years:
            return None
        
        # Create comprehensive balance sheet chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Assets vs Liabilities & Equity', 'Total Assets Trend', 
                          'Debt Levels', 'Equity Growth'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15
        )
        
        # Assets vs Liabilities & Equity (Stacked)
        fig.add_trace(
            go.Bar(x=years, y=liabilities_values, name='Liabilities', 
                  marker_color=COLORS['danger']), row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=years, y=equity_values, name='Equity', 
                  marker_color=COLORS['secondary']), row=1, col=1
        )
        
        # Total Assets Trend
        fig.add_trace(
            go.Scatter(x=years, y=assets_values, name='Total Assets',
                      line=dict(color=COLORS['primary'], width=3),
                      mode='lines+markers', marker=dict(size=8)), row=1, col=2
        )
        
        # Debt Levels
        fig.add_trace(
            go.Bar(x=years, y=debt_values, name='Total Debt',
                  marker_color=COLORS['warning']), row=2, col=1
        )
        
        # Equity Growth
        fig.add_trace(
            go.Scatter(x=years, y=equity_values, name='Shareholders Equity',
                      line=dict(color=COLORS['success'], width=3),
                      mode='lines+markers', marker=dict(size=8)), row=2, col=2
        )
        
        # Update layout for stacked bar in first subplot
        fig.update_layout(barmode='stack')
        
        fig.update_layout(
            height=700,
            title_text="Comprehensive Balance Sheet Analysis (Billions USD)",
            template='plotly_white',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating balance sheet analysis: {str(e)}")
        st.error(f"Error creating balance sheet analysis: {str(e)}")
        return None

@st.cache_data(ttl=600)
def create_cash_flow_analysis(cash_flow: pd.DataFrame) -> Optional[go.Figure]:
    """
    Comprehensive cash flow analysis
    """
    try:
        if cash_flow is None or cash_flow.empty:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get cash flow metrics
        operating_cf = charts._safe_get_metric(cash_flow, FINANCIAL_METRICS['operating_cf'])
        capex = charts._safe_get_metric(cash_flow, FINANCIAL_METRICS['capex'])
        
        # Calculate free cash flow
        free_cf_data = []
        years = []
        operating_values = []
        capex_values = []
        
        for date in operating_cf.index:
            year = str(date)[:4]
            years.append(year)
            
            op_cf = operating_cf[date] / 1e9
            operating_values.append(op_cf)
            
            capex_val = capex[date] / 1e9 if date in capex.index else 0
            capex_values.append(abs(capex_val))  # Make positive for display
            
            # Free cash flow = Operating CF - CapEx
            fcf = op_cf - abs(capex_val)
            free_cf_data.append(fcf)
        
        if not years:
            return None
        
        # Create cash flow analysis chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cash Flow Components', 'Free Cash Flow Trend'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.15
        )
        
        # Cash flow components
        fig.add_trace(
            go.Bar(x=years, y=operating_values, name='Operating CF',
                  marker_color=COLORS['success']), row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=years, y=[-val for val in capex_values], name='Capital Expenditures',
                  marker_color=COLORS['danger']), row=1, col=1
        )
        
        # Free cash flow trend
        fig.add_trace(
            go.Scatter(x=years, y=free_cf_data, name='Free Cash Flow',
                      line=dict(color=COLORS['primary'], width=4),
                      mode='lines+markers', marker=dict(size=10),
                      fill='tonexty'), row=2, col=1
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="Cash Flow Analysis (Billions USD)",
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating cash flow analysis: {str(e)}")
        st.error(f"Error creating cash flow analysis: {str(e)}")
        return None

# Utility function for batch chart creation
@st.cache_data(ttl=600)
def create_financial_charts_batch(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
                                cash_flow: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create multiple financial charts in batch for better performance
    """
    charts = {}
    
    try:
        # Income statement charts
        if income_stmt is not None:
            charts['income_combined'] = create_combined_income_chart(income_stmt)
            charts['growth_analysis'] = create_comprehensive_growth_analysis(income_stmt)
            charts['profitability'] = create_advanced_profitability_analysis(income_stmt)
        
        # Balance sheet charts
        if balance_sheet is not None:
            charts['balance_sheet'] = create_comprehensive_balance_sheet_analysis(balance_sheet)
        
        # Combined analysis
        if income_stmt is not None and balance_sheet is not None:
            charts['dupont'] = create_enhanced_dupont_analysis(income_stmt, balance_sheet)
            charts['health_dashboard'] = create_financial_health_dashboard(
                income_stmt, balance_sheet, cash_flow
            )
        
        # Cash flow charts
        if cash_flow is not None:
            charts['cash_flow'] = create_cash_flow_analysis(cash_flow)
        
        # Filter out None values
        charts = {k: v for k, v in charts.items() if v is not None}
        
    except Exception as e:
        logger.error(f"Error in batch chart creation: {str(e)}")
        st.error(f"Error creating financial charts: {str(e)}")
    
    return charts

@st.cache_data(ttl=600)
def create_historical_ratios_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[go.Figure]:
    """
    Optimized historical ratios chart with enhanced error handling and performance
    """
    try:
        if income_stmt is None or balance_sheet is None:
            return None
        
        charts = FinancialChartsOptimized()
        
        # Get required metrics using optimized method
        revenue_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
        net_income_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
        gross_profit_data = charts._safe_get_metric(income_stmt, FINANCIAL_METRICS['gross_profit'])
        
        assets_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_assets'])
        equity_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
        current_assets_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['current_assets'])
        current_liabilities_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['current_liabilities'])
        debt_data = charts._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_debt'])
        
        # Calculate ratios for each available year
        ratios_data = []
        
        # Find common dates between income statement and balance sheet
        common_dates = set(income_stmt.columns).intersection(set(balance_sheet.columns))
        
        for date in sorted(common_dates, reverse=True):  # Most recent first
            year = str(date)[:4]
            
            # Get values for this year
            revenue = revenue_data[date] if date in revenue_data.index else 0
            net_income = net_income_data[date] if date in net_income_data.index else 0
            gross_profit = gross_profit_data[date] if date in gross_profit_data.index else 0
            total_assets = assets_data[date] if date in assets_data.index else 0
            total_equity = equity_data[date] if date in equity_data.index else 0
            current_assets = current_assets_data[date] if date in current_assets_data.index else 0
            current_liabilities = current_liabilities_data[date] if date in current_liabilities_data.index else 0
            total_debt = debt_data[date] if date in debt_data.index else 0
            
            # Calculate ratios only if we have valid data
            if revenue > 0 and total_assets > 0 and pd.notna(net_income):
                ratio_entry = {'year': year}
                
                # Profitability ratios
                ratio_entry['gross_margin'] = (gross_profit / revenue * 100) if gross_profit and gross_profit > 0 else 0
                ratio_entry['net_margin'] = (net_income / revenue * 100)
                
                # Return ratios
                ratio_entry['roa'] = (net_income / total_assets * 100)
                ratio_entry['roe'] = (net_income / total_equity * 100) if total_equity > 0 else 0
                
                # Liquidity ratios
                ratio_entry['current_ratio'] = (current_assets / current_liabilities) if current_liabilities > 0 else 0
                
                # Leverage ratios
                ratio_entry['debt_to_equity'] = (total_debt / total_equity) if total_equity > 0 else 0
                ratio_entry['debt_to_assets'] = (total_debt / total_assets) if total_assets > 0 else 0
                
                # Asset efficiency ratios
                ratio_entry['asset_turnover'] = (revenue / total_assets) if total_assets > 0 else 0
                
                ratios_data.append(ratio_entry)
        
        if not ratios_data:
            return None
        
        df_ratios = pd.DataFrame(ratios_data)
        
        # Create enhanced subplots for different ratio categories
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Profitability Ratios (%)', 'Return Ratios (%)', 
                'Liquidity & Efficiency', 'Leverage Ratios',
                'Trend Analysis', 'Risk Metrics'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Profitability ratios
        if 'gross_margin' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['gross_margin'], 
                    name='Gross Margin %',
                    line=dict(color=COLORS['primary'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Gross Margin</b><br>Year: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                ), row=1, col=1
            )
        
        if 'net_margin' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['net_margin'], 
                    name='Net Margin %',
                    line=dict(color=COLORS['secondary'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Net Margin</b><br>Year: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                ), row=1, col=1
            )
        
        # 2. Return ratios
        if 'roa' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['roa'], 
                    name='ROA %',
                    line=dict(color=COLORS['accent'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>ROA</b><br>Year: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                ), row=1, col=2
            )
        
        if 'roe' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['roe'], 
                    name='ROE %',
                    line=dict(color=COLORS['danger'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>ROE</b><br>Year: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                ), row=1, col=2
            )
        
        # 3. Liquidity & Efficiency
        if 'current_ratio' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['current_ratio'], 
                    name='Current Ratio',
                    line=dict(color=COLORS['info'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Current Ratio</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ), row=2, col=1
            )
            # Add reference line for healthy current ratio
            fig.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5, 
                         annotation_text="Healthy Level", row=2, col=1)
        
        if 'asset_turnover' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['asset_turnover'], 
                    name='Asset Turnover',
                    line=dict(color=COLORS['success'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Asset Turnover</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ), row=2, col=1
            )
        
        # 4. Leverage ratios
        if 'debt_to_equity' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['debt_to_equity'], 
                    name='Debt/Equity',
                    line=dict(color=COLORS['warning'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Debt/Equity</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ), row=2, col=2
            )
        
        if 'debt_to_assets' in df_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['debt_to_assets'], 
                    name='Debt/Assets',
                    line=dict(color=COLORS['light'], width=3),
                    mode='lines+markers', marker=dict(size=6),
                    hovertemplate='<b>Debt/Assets</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ), row=2, col=2
            )
        
        # 5. Trend Analysis - Combined key metrics
        if all(col in df_ratios.columns for col in ['roe', 'roa', 'current_ratio']):
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['roe'], 
                    name='ROE Trend',
                    line=dict(color=COLORS['danger'], width=2, dash='solid'),
                    mode='lines+markers', marker=dict(size=4)
                ), row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_ratios['year'], y=df_ratios['roa'], 
                    name='ROA Trend',
                    line=dict(color=COLORS['accent'], width=2, dash='dot'),
                    mode='lines+markers', marker=dict(size=4)
                ), row=3, col=1
            )
        
        # 6. Risk Metrics - Volatility of key ratios
        if len(df_ratios) > 2:  # Need at least 3 years for volatility calculation
            risk_metrics = []
            for year in df_ratios['year']:
                # Calculate rolling volatility of ROE (simplified)
                roe_values = df_ratios['roe'].rolling(window=3, min_periods=2).std()
                risk_metrics.append(roe_values.iloc[-1] if not roe_values.empty else 0)
            
            fig.add_trace(
                go.Bar(
                    x=df_ratios['year'], y=risk_metrics,
                    name='ROE Volatility',
                    marker_color=COLORS['warning'],
                    opacity=0.7
                ), row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,  # Increased height for 3 rows
            title_text="Comprehensive Historical Financial Ratios Analysis",
            showlegend=True,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating historical ratios chart: {str(e)}")
        st.error(f"Error creating historical ratios chart: {str(e)}")
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
    
@st.cache_data(ttl=600)
def create_historical_performance_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
                                      cash_flow: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create comprehensive historical performance charts for all statements
    
    Args:
        income_stmt (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        cash_flow (pd.DataFrame): Cash flow data
        
    Returns:
        dict: Dictionary of chart figures
    """
    try:
        charts = {}
        charts_helper = FinancialChartsOptimized()
        
        # Income Statement Charts
        if income_stmt is not None and not income_stmt.empty:
            # Revenue Evolution
            revenue_data = charts_helper._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
            if not revenue_data.empty:
                years, values = charts_helper._prepare_yearly_data(revenue_data)
                
                charts['revenue'] = go.Figure()
                charts['revenue'].add_trace(go.Bar(
                    x=years, y=values,
                    name='Revenue',
                    marker_color=COLORS['primary'],
                    hovertemplate='<b>Revenue</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['revenue'].update_layout(
                    **charts_helper._create_base_layout('Historical Revenue Evolution')
                )
            
            # Net Income Evolution
            income_data = charts_helper._safe_get_metric(income_stmt, FINANCIAL_METRICS['net_income'])
            if not income_data.empty:
                years, values = charts_helper._prepare_yearly_data(income_data)
                
                charts['net_income'] = go.Figure()
                charts['net_income'].add_trace(go.Bar(
                    x=years, y=values,
                    name='Net Income',
                    marker_color=[COLORS['secondary'] if v >= 0 else COLORS['danger'] for v in values],
                    hovertemplate='<b>Net Income</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['net_income'].update_layout(
                    **charts_helper._create_base_layout('Historical Net Income')
                )
                # Add zero line for reference
                charts['net_income'].add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            # Profitability Margins Over Time
            revenue_data = charts_helper._safe_get_metric(income_stmt, FINANCIAL_METRICS['revenue'])
            gross_profit_data = charts_helper._safe_get_metric(income_stmt, FINANCIAL_METRICS['gross_profit'])
            operating_income_data = charts_helper._safe_get_metric(income_stmt, FINANCIAL_METRICS['operating_income'])
            
            if not revenue_data.empty:
                margin_data = []
                for date in revenue_data.index:
                    year = str(date)[:4]
                    revenue = revenue_data[date]
                    
                    if revenue > 0:
                        margins = {'year': year}
                        
                        if date in gross_profit_data.index:
                            margins['gross_margin'] = (gross_profit_data[date] / revenue) * 100
                        
                        if date in operating_income_data.index:
                            margins['operating_margin'] = (operating_income_data[date] / revenue) * 100
                        
                        if date in income_data.index:
                            margins['net_margin'] = (income_data[date] / revenue) * 100
                        
                        margin_data.append(margins)
                
                if margin_data:
                    df_margins = pd.DataFrame(margin_data)
                    charts['margins'] = go.Figure()
                    
                    if 'gross_margin' in df_margins.columns:
                        charts['margins'].add_trace(go.Scatter(
                            x=df_margins['year'], y=df_margins['gross_margin'],
                            name='Gross Margin', line=dict(color=COLORS['primary'], width=3),
                            mode='lines+markers'
                        ))
                    
                    if 'operating_margin' in df_margins.columns:
                        charts['margins'].add_trace(go.Scatter(
                            x=df_margins['year'], y=df_margins['operating_margin'],
                            name='Operating Margin', line=dict(color=COLORS['accent'], width=3),
                            mode='lines+markers'
                        ))
                    
                    if 'net_margin' in df_margins.columns:
                        charts['margins'].add_trace(go.Scatter(
                            x=df_margins['year'], y=df_margins['net_margin'],
                            name='Net Margin', line=dict(color=COLORS['secondary'], width=3),
                            mode='lines+markers'
                        ))
                    
                    layout = charts_helper._create_base_layout(
                        'Historical Profit Margins', yaxis_title='Margin (%)'
                    )
                    layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    charts['margins'].update_layout(**layout)
        
        # Balance Sheet Charts
        if balance_sheet is not None and not balance_sheet.empty:
            # Total Assets Evolution
            assets_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_assets'])
            if not assets_data.empty:
                years, values = charts_helper._prepare_yearly_data(assets_data)
                
                charts['total_assets'] = go.Figure()
                charts['total_assets'].add_trace(go.Scatter(
                    x=years, y=values,
                    name='Total Assets',
                    line=dict(color=COLORS['accent'], width=4),
                    mode='lines+markers',
                    marker=dict(size=8),
                    fill='tonexty',
                    hovertemplate='<b>Total Assets</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['total_assets'].update_layout(
                    **charts_helper._create_base_layout('Historical Total Assets')
                )
            
            # Debt Evolution
            debt_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_debt'])
            if not debt_data.empty:
                years, values = charts_helper._prepare_yearly_data(debt_data)
                
                charts['total_debt'] = go.Figure()
                charts['total_debt'].add_trace(go.Bar(
                    x=years, y=values,
                    name='Total Debt',
                    marker_color=COLORS['danger'],
                    hovertemplate='<b>Total Debt</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['total_debt'].update_layout(
                    **charts_helper._create_base_layout('Historical Total Debt')
                )
            
            # Shareholders Equity Evolution
            equity_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
            if not equity_data.empty:
                years, values = charts_helper._prepare_yearly_data(equity_data)
                
                charts['equity'] = go.Figure()
                charts['equity'].add_trace(go.Scatter(
                    x=years, y=values,
                    name='Shareholders Equity',
                    line=dict(color=COLORS['info'], width=4),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate='<b>Equity</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['equity'].update_layout(
                    **charts_helper._create_base_layout('Historical Shareholders Equity')
                )
        
        # Cash Flow Charts
        if cash_flow is not None and not cash_flow.empty:
            # Operating Cash Flow
            ocf_data = charts_helper._safe_get_metric(cash_flow, FINANCIAL_METRICS['operating_cf'])
            if not ocf_data.empty:
                years, values = charts_helper._prepare_yearly_data(ocf_data)
                
                charts['operating_cf'] = go.Figure()
                charts['operating_cf'].add_trace(go.Bar(
                    x=years, y=values,
                    name='Operating Cash Flow',
                    marker_color=[COLORS['success'] if v >= 0 else COLORS['danger'] for v in values],
                    hovertemplate='<b>Operating CF</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                ))
                charts['operating_cf'].update_layout(
                    **charts_helper._create_base_layout('Historical Operating Cash Flow')
                )
                charts['operating_cf'].add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            # Free Cash Flow
            capex_data = charts_helper._safe_get_metric(cash_flow, FINANCIAL_METRICS['capex'])
            if not ocf_data.empty and not capex_data.empty:
                # Calculate free cash flow
                common_dates = ocf_data.index.intersection(capex_data.index)
                if len(common_dates) > 0:
                    fcf_data = ocf_data[common_dates] + capex_data[common_dates]  # CapEx is typically negative
                    years, values = charts_helper._prepare_yearly_data(fcf_data)
                    
                    charts['free_cf'] = go.Figure()
                    charts['free_cf'].add_trace(go.Scatter(
                        x=years, y=values,
                        name='Free Cash Flow',
                        line=dict(color=COLORS['light'], width=4),
                        mode='lines+markers',
                        marker=dict(size=8),
                        fill='tonexty',
                        hovertemplate='<b>Free CF</b><br>Year: %{x}<br>Amount: $%{y:.2f}B<extra></extra>'
                    ))
                    charts['free_cf'].update_layout(
                        **charts_helper._create_base_layout('Historical Free Cash Flow')
                    )
                    charts['free_cf'].add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Filter out None values
        charts = {k: v for k, v in charts.items() if v is not None}
        
        return charts
        
    except Exception as e:
        logger.error(f"Error creating historical performance charts: {str(e)}")
        st.error(f"Error creating historical performance charts: {str(e)}")
        return {}

@st.cache_data(ttl=600)
def create_debt_analysis_chart(balance_sheet: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create comprehensive debt analysis chart with multiple debt metrics
    
    Args:
        balance_sheet (pd.DataFrame): Balance sheet data
        
    Returns:
        plotly.graph_objects.Figure: Debt analysis chart
    """
    try:
        if balance_sheet is None or balance_sheet.empty:
            return None
        
        charts_helper = FinancialChartsOptimized()
        
        # Get debt-related metrics
        total_debt_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_debt'])
        total_assets_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['total_assets'])
        equity_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['stockholders_equity'])
        current_liabilities_data = charts_helper._safe_get_metric(balance_sheet, FINANCIAL_METRICS['current_liabilities'])
        
        # Try to get long-term debt separately
        long_term_debt_data = charts_helper._safe_get_metric(balance_sheet, ['Long Term Debt'])
        
        # Calculate debt metrics
        debt_analysis_data = []
        
        for date in total_assets_data.index:
            year = str(date)[:4]
            
            total_assets = total_assets_data[date] if date in total_assets_data.index else 0
            total_debt = total_debt_data[date] if date in total_debt_data.index else 0
            equity = equity_data[date] if date in equity_data.index else 0
            current_liab = current_liabilities_data[date] if date in current_liabilities_data.index else 0
            long_term_debt = long_term_debt_data[date] if date in long_term_debt_data.index else 0
            
            if total_assets > 0:
                debt_entry = {
                    'year': year,
                    'total_debt_billions': total_debt / 1e9,
                    'long_term_debt_billions': long_term_debt / 1e9,
                    'current_liabilities_billions': current_liab / 1e9,
                    'debt_to_assets': (total_debt / total_assets) * 100 if total_debt > 0 else 0,
                    'debt_to_equity': (total_debt / equity) if equity > 0 and total_debt > 0 else 0,
                    'equity_ratio': (equity / total_assets) * 100 if equity > 0 else 0
                }
                debt_analysis_data.append(debt_entry)
        
        if not debt_analysis_data:
            return None
        
        df_debt = pd.DataFrame(debt_analysis_data)
        
        # Create comprehensive debt analysis chart
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Debt Composition (Billions USD)', 'Debt-to-Assets Ratio (%)',
                'Debt-to-Equity Ratio', 'Equity Ratio (%)',
                'Debt Trend Analysis', 'Leverage Risk Assessment'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12
        )
        
        # 1. Debt Composition (Stacked Bar)
        if 'long_term_debt_billions' in df_debt.columns:
            fig.add_trace(
                go.Bar(x=df_debt['year'], y=df_debt['long_term_debt_billions'], 
                      name='Long-term Debt', marker_color=COLORS['danger']),
                row=1, col=1
            )
        
        if 'current_liabilities_billions' in df_debt.columns:
            fig.add_trace(
                go.Bar(x=df_debt['year'], y=df_debt['current_liabilities_billions'], 
                      name='Current Liabilities', marker_color=COLORS['warning']),
                row=1, col=1
            )
        
        # 2. Debt-to-Assets Ratio
        fig.add_trace(
            go.Scatter(x=df_debt['year'], y=df_debt['debt_to_assets'], 
                      name='Debt/Assets %', line=dict(color=COLORS['primary'], width=3),
                      mode='lines+markers', marker=dict(size=8)),
            row=1, col=2
        )
        # Add reference lines for debt ratios
        fig.add_hline(y=30, line_dash="dash", line_color="orange", opacity=0.5, 
                     annotation_text="Moderate (30%)", row=1, col=2)
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, 
                     annotation_text="High (50%)", row=1, col=2)
        
        # 3. Debt-to-Equity Ratio
        fig.add_trace(
            go.Scatter(x=df_debt['year'], y=df_debt['debt_to_equity'], 
                      name='Debt/Equity', line=dict(color=COLORS['accent'], width=3),
                      mode='lines+markers', marker=dict(size=8)),
            row=2, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="orange", opacity=0.5, 
                     annotation_text="1:1 Ratio", row=2, col=1)
        
        # 4. Equity Ratio
        fig.add_trace(
            go.Scatter(x=df_debt['year'], y=df_debt['equity_ratio'], 
                      name='Equity Ratio %', line=dict(color=COLORS['secondary'], width=3),
                      mode='lines+markers', marker=dict(size=8)),
            row=2, col=2
        )
        
        # 5. Debt Trend Analysis (Total Debt)
        fig.add_trace(
            go.Scatter(x=df_debt['year'], y=df_debt['total_debt_billions'], 
                      name='Total Debt Trend', line=dict(color=COLORS['danger'], width=4),
                      mode='lines+markers', marker=dict(size=10), fill='tonexty'),
            row=3, col=1
        )
        
        # 6. Risk Assessment (Combined metrics)
        # Create a simple risk score based on debt ratios
        risk_scores = []
        for _, row in df_debt.iterrows():
            risk_score = 0
            if row['debt_to_assets'] > 50:
                risk_score += 3
            elif row['debt_to_assets'] > 30:
                risk_score += 2
            elif row['debt_to_assets'] > 15:
                risk_score += 1
            
            if row['debt_to_equity'] > 2:
                risk_score += 3
            elif row['debt_to_equity'] > 1:
                risk_score += 2
            elif row['debt_to_equity'] > 0.5:
                risk_score += 1
            
            risk_scores.append(risk_score)
        
        # Color code risk levels
        risk_colors = []
        for score in risk_scores:
            if score >= 5:
                risk_colors.append(COLORS['danger'])  # High risk
            elif score >= 3:
                risk_colors.append(COLORS['warning'])  # Medium risk
            else:
                risk_colors.append(COLORS['success'])  # Low risk
        
        fig.add_trace(
            go.Bar(x=df_debt['year'], y=risk_scores, 
                  name='Leverage Risk Score', marker_color=risk_colors),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Comprehensive Debt Analysis",
            template='plotly_white',
            showlegend=True,
            barmode='stack')
    except Exception as e:
        logger.error(f"Error creating debt analysis chart: {str(e)}")
        st.error(f"Error creating debt analysis chart: {str(e)}")
        return None