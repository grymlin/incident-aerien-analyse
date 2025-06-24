# geo_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, List

# Country to continent mapping for regional analysis
CONTINENT_MAPPING = {
    'Belgium' : 'Europe','USA' : 'North America','United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Colombia': 'South America', 'Chile': 'South America',
    'United Kingdom': 'Europe', 'Germany': 'Europe','U.K.':'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
    'Russia': 'Europe', 'Netherlands': 'Europe', 'Switzerland': 'Europe', 'Norway': 'Europe', 'Sweden': 'Europe',
    'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 'Indonesia': 'Asia', 'Pakistan': 'Asia', 'Iran': 'Asia',
    'Thailand': 'Asia', 'Philippines': 'Asia', 'Malaysia': 'Asia', 'South Korea': 'Asia', 'Turkey': 'Asia',
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania','D.R. Congo' : 'Africa', 'Libya':'Africa',
    'Nigeria': 'Africa', 'Egypt': 'Africa', 'South Africa': 'Africa', 'Kenya': 'Africa', 'Morocco': 'Africa',
    'Ethiopia': 'Africa', 'Algeria': 'Africa', 'Ghana': 'Africa', 'Angola': 'Africa'
}

def prepare_geo_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare comprehensive country-level accident analysis"""
    if 'country' not in df.columns:
        return None
    
    # Clean and prepare data
    df_clean = df[df['country'].notna()].copy()
    df_clean['fatalities'] = pd.to_numeric(df_clean['fatalities'], errors='coerce').fillna(0)
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
    
    # Add continent mapping
    df_clean['continent'] = df_clean['country'].map(CONTINENT_MAPPING).fillna('Other')
    
    # Calculate comprehensive metrics per country
    geo_stats = df_clean.groupby('country').agg({
        'fatalities': ['sum', 'mean', 'count'],
        'year': ['min', 'max', 'count'],
        'cat': lambda x: (x == 'A1').sum() if 'cat' in df_clean.columns else 0,
        'continent': 'first'
    }).reset_index()
    
    # Flatten column names
    geo_stats.columns = [
        'country', 'total_fatalities', 'avg_fatalities_per_accident', 'total_accidents',
        'first_accident_year', 'last_accident_year', 'years_with_accidents',
        'hull_loss_accidents', 'continent'
    ]
    
    # Calculate additional metrics
    geo_stats['accident_span_years'] = geo_stats['last_accident_year'] - geo_stats['first_accident_year']
    geo_stats['hull_loss_rate'] = geo_stats['hull_loss_accidents'] / geo_stats['total_accidents']
    geo_stats['fatality_rate'] = geo_stats['total_fatalities'] / geo_stats['total_accidents']
    
    # Calculate safety score (lower is better)
    geo_stats['safety_score'] = (
        geo_stats['fatality_rate'] * 0.4 +
        geo_stats['hull_loss_rate'] * 0.3 +
        (geo_stats['total_accidents'] / geo_stats['accident_span_years'].replace(0, 1)) * 0.3
    ).round(2)
    return geo_stats

def plot_choropleth(geo_df: pd.DataFrame, metric: str = 'total_accidents'):
    """Create interactive choropleth map"""
    metric_labels = {
        'total_accidents': 'Total Accidents',
        'total_fatalities': 'Total Fatalities',
        'fatality_rate': 'Avg Fatalities per Accident',
        'hull_loss_rate': 'Hull Loss Rate (%)',
        'safety_score': 'Safety Risk Score'
    }
    
    # Prepare data for visualization
    plot_data = geo_df.copy()
    if metric == 'hull_loss_rate':
        plot_data[metric] = plot_data[metric] * 100  # Convert to percentage
    
    # Choose appropriate color scale
    color_scales = {
        'total_accidents': 'Reds',
        'total_fatalities': 'OrRd',
        'fatality_rate': 'YlOrRd',
        'hull_loss_rate': 'Oranges',
        'safety_score': 'RdYlGn_r'
    }
    
    fig = px.choropleth(
        plot_data,
        locations='country',
        locationmode='country names',
        color=metric,
        hover_name='country',
        hover_data={
            'continent': True,
            'total_accidents': True,
            'total_fatalities': True,
            'fatality_rate': ':.1f',
            'hull_loss_rate': ':.1%',
            'safety_score': ':.2f'
        },
        color_continuous_scale=color_scales.get(metric, 'Viridis'),
        projection='natural earth',
        title=f'<b>Global Aviation Safety: {metric_labels[metric]}</b>',
        height=600
    )
    
    # styling
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="white",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="lightblue",
        projection_scale=1
    )
    
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        coloraxis_colorbar=dict(
            title=metric_labels[metric],
            thickness=15,
            len=0.7
        )
    )
    
    return fig

def plot_continental_analysis(geo_df: pd.DataFrame):
    """Comprehensive continental comparison"""
    # Continental statistics
    continent_stats = geo_df.groupby('continent').agg({
        'total_accidents': 'sum',
        'total_fatalities': 'sum',
        'country': 'count',
        'fatality_rate': 'mean',
        'hull_loss_rate': 'mean',
        'safety_score': 'mean'
    }).reset_index()
    
    continent_stats.columns = [
        'continent', 'total_accidents', 'total_fatalities', 'countries_with_accidents',
        'avg_fatality_rate', 'avg_hull_loss_rate', 'avg_safety_score'
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Total Accidents by Continent',
            'Average Fatality Rate by Continent',
            'Hull Loss Rate by Continent',
            'Countries with Accidents'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Continental accidents
    fig.add_trace(
        go.Bar(
            x=continent_stats['continent'],
            y=continent_stats['total_accidents'],
            name='Accidents',
            marker_color='lightcoral'
        ),
        row=1, col=1
    )
    
    # Average fatality rate
    fig.add_trace(
        go.Bar(
            x=continent_stats['continent'],
            y=continent_stats['avg_fatality_rate'],
            name='Avg Fatalities',
            marker_color='orange'
        ),
        row=1, col=2
    )
    
    # Hull loss rate
    fig.add_trace(
        go.Bar(
            x=continent_stats['continent'],
            y=continent_stats['avg_hull_loss_rate'] * 100,
            name='Hull Loss %',
            marker_color='red'
        ),
        row=2, col=1
    )
    
    # Countries count
    fig.add_trace(
        go.Bar(
            x=continent_stats['continent'],
            y=continent_stats['countries_with_accidents'],
            name='Countries',
            marker_color='steelblue'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Continental Aviation Safety Analysis",
        title_x=0.5
    )
    
    return fig

def plot_country_ranking(geo_df: pd.DataFrame, metric: str, top_n: int = 15):
    """Country ranking visualization"""
    metric_labels = {
        'total_accidents': 'Total Accidents',
        'total_fatalities': 'Total Fatalities',
        'fatality_rate': 'Avg Fatalities per Accident',
        'safety_score': 'Safety Risk Score'
    }
    
    # Get top countries
    top_countries = geo_df.nlargest(top_n, metric)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars with color gradient
    fig.add_trace(go.Bar(
        y=top_countries['country'][::-1],  # Reverse for better visualization
        x=top_countries[metric][::-1],
        orientation='h',
        marker=dict(
            color=top_countries[metric][::-1],
            colorscale='RdYlBu_r' if metric == 'safety_score' else 'Reds',
            showscale=True,
            colorbar=dict(title=metric_labels[metric])
        ),
        text=top_countries[metric][::-1].round(1),
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            f'{metric_labels[metric]}: %{{x}}<br>' +
            'Continent: %{customdata}<br>' +
            '<extra></extra>'
        ),
        customdata=top_countries['continent'][::-1]
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Countries by {metric_labels[metric]}',
        xaxis_title=metric_labels[metric],
        yaxis_title='Country',
        height=600,
        margin=dict(l=150, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_safety_bubble_chart(geo_df: pd.DataFrame):
    """Bubble chart showing accidents vs fatalities with safety context"""
    # Filter for meaningful data (countries with multiple accidents)
    significant_countries = geo_df[geo_df['total_accidents'] >= 2].copy()
    
    fig = px.scatter(
        significant_countries,
        x='total_accidents',
        y='total_fatalities',
        size='fatality_rate',
        color='safety_score',
        hover_name='country',
        hover_data={
            'continent': True,
            'hull_loss_rate': ':.1%',
            'accident_span_years': True
        },
        color_continuous_scale='RdYlGn_r',
        size_max=50,
        title='Country Safety Profile: Accidents vs Fatalities',
        labels={
            'total_accidents': 'Total Accidents',
            'total_fatalities': 'Total Fatalities',
            'safety_score': 'Safety Risk Score'
        }
    )
    
    # Add trend line
    if len(significant_countries) > 2:
        fig.add_trace(
            go.Scatter(
                x=significant_countries['total_accidents'],
                y=significant_countries['total_accidents'] * significant_countries['fatality_rate'].median(),
                mode='lines',
                name='Median Fatality Trend',
                line=dict(dash='dash', color='gray'),
                hovertemplate='<extra></extra>'
            )
        )
    
    fig.update_layout(
        xaxis_title='Total Accidents',
        yaxis_title='Total Fatalities',
        height=600
    )
    
    return fig

def create_geographic_summary_table(geo_df: pd.DataFrame):
    """Create comprehensive summary statistics table"""
    # Overall statistics
    total_countries = len(geo_df)
    total_accidents = geo_df['total_accidents'].sum()
    total_fatalities = geo_df['total_fatalities'].sum()
    
    # Continental breakdown
    continent_summary = geo_df.groupby('continent').agg({
        'total_accidents': 'sum',
        'total_fatalities': 'sum',
        'country': 'count'
    }).round(1)
    
    # Top performers (lowest safety scores)
    safest_countries = geo_df[geo_df['total_accidents'] >= 2].nsmallest(5, 'safety_score')
    riskiest_countries = geo_df[geo_df['total_accidents'] >= 2].nlargest(5, 'safety_score')
    
    return {
        'overview': {
            'Countries with accidents': total_countries,
            'Total accidents recorded': total_accidents,
            'Total fatalities': total_fatalities,
            'Average accidents per country': round(total_accidents / total_countries, 1)
        },
        'continent_summary': continent_summary,
        'safest_countries': safest_countries[['country', 'total_accidents', 'safety_score']],
        'riskiest_countries': riskiest_countries[['country', 'total_accidents', 'safety_score']]
    }

def display_geo_analysis(df: pd.DataFrame):
    """Main enhanced geographic analysis display"""

    # Prepare data
    geo_df = prepare_geo_data(df)
    if geo_df is None or geo_df.empty:
        st.warning("No geographic data available for analysis")
        return
    # Summary statistics
    st.subheader("Geographic Summary Statistics")
    summary_data = create_geographic_summary_table(geo_df)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries Analyzed", summary_data['overview']['Countries with accidents'])
    with col2:
        st.metric("Total Accidents", summary_data['overview']['Total accidents recorded'])
    with col3:
        st.metric("Total Fatalities", summary_data['overview']['Total fatalities'])
    with col4:
        st.metric("Avg per Country", summary_data['overview']['Average accidents per country'])

    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        map_metric = st.selectbox(
            "Map Visualization Metric:",
            ['total_accidents', 'total_fatalities', 'fatality_rate', 'hull_loss_rate', 'safety_score'],
            format_func=lambda x: {
                'total_accidents': 'Total Accidents',
                'total_fatalities': 'Total Fatalities', 
                'fatality_rate': 'Fatality Rate',
                'hull_loss_rate': 'Hull Loss Rate',
                'safety_score': 'Safety Risk Score'
            }[x]
        )
    
    with col2:
        ranking_metric = st.selectbox(
            "Country Ranking Metric:",
            ['total_accidents', 'total_fatalities', 'fatality_rate', 'safety_score'],
            format_func=lambda x: {
                'total_accidents': 'Total Accidents',
                'total_fatalities': 'Total Fatalities',
                'fatality_rate': 'Fatality Rate', 
                'safety_score': 'Safety Risk Score'
            }[x]
        )
    
    with col3:
        top_n = st.slider("Show top N countries:", 5, 25, 15)
    
    # Choropleth map
    st.subheader("Global Safety Heatmap")
    map_fig = plot_choropleth(geo_df, map_metric)
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Continental analysis
    st.subheader("Continental Comparison")
    continent_fig = plot_continental_analysis(geo_df)
    st.plotly_chart(continent_fig, use_container_width=True)
    
    st.subheader("Country Rankings")
    ranking_fig = plot_country_ranking(geo_df, ranking_metric, top_n)
    st.plotly_chart(ranking_fig, use_container_width=True)
            

    # Continental summary
    with st.expander("Continental Breakdown"):
        st.dataframe(summary_data['continent_summary'])
    
    # Export options
    st.subheader("Data Export")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download Geographic Analysis (CSV)",
            data=geo_df.to_csv(index=False),
            file_name="geographic_safety_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            "Download Continental Summary (CSV)",
            data=summary_data['continent_summary'].to_csv(),
            file_name="continental_summary.csv",
            mime="text/csv"
        )