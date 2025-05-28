# Add these imports at the top of your streamlit_app.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import numpy as np
from pyxlsb import open_workbook
import warnings


"""
    Creation and modification:
    Creation date               : 28/05/2025

    Creation date               : 28/05/2025

    @author                     : Rym Otsmane

    
"""

def safe_lowess_smoothing(df, x_col, y_col, frac=0.3, min_points=10):
    """
    Apply LOWESS smoothing to data in df[x_col] and df[y_col] safely.

    Parameters:
    - df: pandas DataFrame containing the data
    - x_col: str, column name for independent variable
    - y_col: str, column name for dependent variable
    - frac: float, LOWESS smoothing parameter (between 0 and 1)
    - min_points: int, minimum number of valid points required to run LOWESS

    Returns:
    - smoothed: np.ndarray of shape (n, 2) with columns [x_smoothed, y_smoothed]
      or None if not enough data
    """

    # 1. Remove NaN and infinite values
    mask = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    cleaned = df.loc[mask]

    # 2. Drop duplicate x values (optional but helps LOWESS)
    cleaned = cleaned.drop_duplicates(subset=[x_col])

    # 3. Sort by x_col (LOWESS expects sorted x for sensible output)
    cleaned = cleaned.sort_values(by=x_col)

    # 4. Check enough data points
    if len(cleaned) < min_points:
        # Not enough data to smooth
        return None

    # 5. Run LOWESS with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smoothed = lowess(
            endog=cleaned[y_col],
            exog=cleaned[x_col],
            frac=frac,
            it=0,
            delta=0.0,
        )

    return smoothed

def plot_fatalities_timeline(df):
    st.subheader("Fatalities Timeline by Operator")
    
    # Get top 10 operators for dropdown
    top_operators = df['operator'].value_counts().nlargest(10).index.tolist()
    selected_operators = st.multiselect(
        "Select operators to compare:", 
        options=top_operators,
        default=top_operators[:3]
    )
    
    if selected_operators:
        fig = px.line(
            df[df['operator'].isin(selected_operators)],
            x='year', 
            y='fatalities',
            color='operator',
            markers=True,
            title='Fatalities Over Time by Operator',
            hover_data=['type', 'country', 'cat']
        )
        fig.update_layout(
            hovermode='x unified',
            xaxis_title='Year',
            yaxis_title='Number of Fatalities'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one operator")

def plot_accident_categories(df):
    st.subheader("Accident Severity Distribution")
    
    # Group by accident category
    cat_stats = df.groupby('cat').agg({
        'fatalities': 'sum',
        'type': 'count'
    }).rename(columns={'type': 'accident_count'}).reset_index()
    
    # Interactive selector
    col1, col2 = st.columns([3, 1])
    with col1:
        view = st.radio(
            "View by:",
            ["Fatalities Count", "Accident Frequency"],
            horizontal=True
        )
    
    with col2:
        show_table = st.checkbox("Show data table")
    
    fig = px.pie(
        cat_stats,
        values='fatalities' if view == "Fatalities Count" else 'accident_count',
        names='cat',
        hole=0.3,
        title=f"Accidents by Category - {view}",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if show_table:
        st.dataframe(cat_stats.sort_values(
            'fatalities' if view == "Fatalities Count" else 'accident_count', 
            ascending=False
        ))

def plot_manufacturer_safety(df):
    st.subheader("Manufacturer Safety Profile")
    
    # Calculate key metrics
    mfg_stats = df.groupby('manufacturer').agg({
        'fatalities': ['sum', 'mean', 'count'],
        'year': ['min', 'max']
    }).droplevel(0, axis=1)
    
    mfg_stats.columns = ['total_fatalities', 'avg_fatalities', 'accident_count', 'first_year', 'last_year']
    mfg_stats = mfg_stats.reset_index()
    mfg_stats['service_years'] = mfg_stats['last_year'] - mfg_stats['first_year']
    
    # Interactive controls
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox(
            "Compare by:",
            ['total_fatalities', 'avg_fatalities', 'accident_count'],
            key='mfg_metric'
        )
    with col2:
        min_accidents = st.slider(
            "Minimum accidents to include:",
            min_value=5,
            max_value=50,
            value=10
        )
    
    filtered = mfg_stats[mfg_stats['accident_count'] >= min_accidents]
    
    if not filtered.empty:
        fig = px.bar(
            filtered.sort_values(metric, ascending=False).head(15),
            x='manufacturer',
            y=metric,
            color='service_years',
            title=f"Top Manufacturers by {metric.replace('_', ' ').title()}",
            hover_data=['accident_count', 'service_years'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Manufacturer',
            yaxis_title=metric.replace('_', ' ').title()
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No manufacturers meet the selected criteria")

def plot_age_impact(df):
    from datetime import datetime
    import plotly.express as px
    import numpy as np
    import streamlit as st

    st.subheader("Aircraft Age Impact Analysis")

    if df.empty:
        st.warning("Dataset is empty.")
        return

    current_year = datetime.now().year
    type_first_year = df.groupby('type')['year'].min().reset_index()
    type_first_year.columns = ['type', 'first_year']
    df = df.merge(type_first_year, on='type', how='left')
    df['aircraft_age'] = df['year'] - df['first_year']

    col1, col2 = st.columns(2)
    with col1:
        age_range = st.slider("Aircraft age range (years):", 0, 50, (0, 30))
    with col2:
        show_outliers = st.checkbox("Show outlier accidents")

    filtered = df[(df['aircraft_age'] >= age_range[0]) & (df['aircraft_age'] <= age_range[1])]

    if not show_outliers and not filtered.empty:
        fatal_threshold = filtered['fatalities'].quantile(0.99)
        filtered = filtered[filtered['fatalities'] <= fatal_threshold]

    filtered = filtered.dropna(subset=['aircraft_age', 'fatalities'])
    filtered = filtered[np.isfinite(filtered['aircraft_age']) & np.isfinite(filtered['fatalities'])]

    if filtered.empty:
        st.warning("No data available after filtering.")
        return

    fig = px.scatter(
        filtered,
        x='aircraft_age',
        y='fatalities',
        color='manufacturer',
        size='fatalities',
        hover_name='type',
        hover_data=['operator', 'year', 'country'],
        title="Fatalities by Aircraft Age",
        labels={'aircraft_age': 'Aircraft Age (years)', 'fatalities': 'Fatalities'}
    )

    lowess_result = safe_lowess_smoothing(filtered, 'aircraft_age', 'fatalities')
    if lowess_result is not None:
        fig.add_scatter(
            x=lowess_result['aircraft_age'],
            y=lowess_result['lowess'],
            mode='lines',
            name='LOWESS Trendline',
            line=dict(color='black', width=2, dash='dash')
        )
    else:
        st.info("LOWESS trendline skipped due to insufficient or unstable data.")

    st.plotly_chart(fig, use_container_width=True)

def plot_time_heatmap(df):
    st.subheader("Temporal Accident Patterns")
    
    # Ensure we have year data and it's clean
    if 'year' not in df.columns:
        st.warning("Year information not available for temporal analysis")
        return
    
    # Clean year data - convert to numeric and drop NA
    df = df.copy()  # Avoid modifying original dataframe
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    
    if df.empty:
        st.warning("No valid year data available")
        return
    
    # Create decade bins (only for valid years)
    df['year'] = df['year'].astype(int)  # Now safe after NA removal
    df['decade'] = (df['year'] // 10) * 10
    
    # Interactive controls
    col1, col2 = st.columns([2, 3])
    with col1:
        time_group = st.radio(
            "Group by:",
            ["Year", "Decade"],
            horizontal=True
        )
        
        show_as = st.radio(
            "Display as:",
            ["Absolute Count", "Normalized"],
            help="Normalized shows percentage within each time period"
        )
    
    # Prepare data based on selections
    time_col = 'decade' if time_group == "Decade" else 'year'
    heat_data = df.groupby([time_col, 'cat']).size().unstack()
    
    if show_as == "Normalized":
        heat_data = heat_data.div(heat_data.sum(axis=1), axis=0) * 100
    
    # Create the heatmap
    fig = px.imshow(
        heat_data,
        labels=dict(
            x="Accident Category",
            y=time_group,
            color="% of Accidents" if show_as == "Normalized" else "Number of Accidents"
        ),
        title=f"Accidents by {time_group} and Category ({len(df)} accidents)",
        color_continuous_scale='YlOrRd',
        aspect="auto",
        text_auto=True if show_as == "Absolute Count" else ".1f"
    )
    
    # Custom hover template
    hover_template = (
        f"{time_group}: %{{y}}<br>"
        "Category: %{x}<br>"
    )
    if show_as == "Normalized":
        hover_template += "Percentage: %{z:.1f}%"
    else:
        hover_template += "Count: %{z}"
    
    fig.update_traces(hovertemplate=hover_template)
    
    # Adjust layout
    fig.update_layout(
        xaxis_title="Accident Category",
        yaxis_title=time_group,
        coloraxis_colorbar_title="% of Accidents" if show_as == "Normalized" else "Count"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_summary_stats(df):
    """Generate summary statistics for download"""
    if df.empty:
        return pd.DataFrame()
    
    # Basic counts
    total_accidents = len(df)
    total_fatalities = df['fatalities'].sum()
    
    # Time statistics
    years = df['year'].dropna()
    year_range = f"{int(years.min())}-{int(years.max())}" if not years.empty else "N/A"
    
    # Operator statistics
    operators = df['operator'].value_counts()
    
    # Aircraft statistics
    manufacturers = df['manufacturer'].value_counts()
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'Category': ['Accidents', 'Fatalities', 'Time Period', 
                    'Operators', 'Aircraft', 'Safety Metrics'],
        'Metric': [
            f"Total: {total_accidents}",
            f"Total: {total_fatalities}\nAvg per accident: {df['fatalities'].mean():.1f}",
            f"Range: {year_range}\nMost recent: {int(years.max()) if not years.empty else 'N/A'}",
            f"Top operator: {operators.index[0] if not operators.empty else 'N/A'}\nTotal operators: {len(operators)}",
            f"Top manufacturer: {manufacturers.index[0] if not manufacturers.empty else 'N/A'}\nTotal types: {df['type'].nunique()}",
            f"Hull loss rate: {df['cat'].eq('A1').mean():.1%}\nFatal accident rate: {(df['fatalities'] > 0).mean():.1%}"
        ]
    })
    
    return summary

def get_download_section(df, filtered_df=None):
    """ download section """
    if filtered_df is None:
        filtered_df = df
        
    st.markdown("---")
    st.subheader("Data Export Center")
    
    # Display summary preview
    with st.expander("Preview Summary Statistics", expanded=True):
        summary = generate_summary_stats(filtered_df)
        if not summary.empty:
            st.table(summary)
        else:
            st.warning("No data available for summary")
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Current View**")
        st.download_button(
            label="CSV (Filtered Data)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name=f"aviation_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            help="Download currently filtered data in CSV format"
        )
    
    with col2:
        st.markdown("**Analysis Results**")
        st.download_button(
            label="Excel",
            data=create_excel_export(
                generate_summary_stats(filtered_df),
                filtered_df,
                generate_safety_metrics(filtered_df)
            ),
            file_name="aviation_analysis_package.xlsx",
            help="Download complete analysis package with multiple sheets"
        )
    
    with col3:
        st.markdown("**Complete Data**")
        st.download_button(
            label="JSON (Full Dataset)",
            data=df.to_json(orient='records', indent=2),
            file_name="aviation_complete_dataset.json",
            help="Download complete dataset in JSON format"
        )

def generate_safety_metrics(df):
    """Generate detailed safety metrics"""
    if df.empty:
        return pd.DataFrame()
    
    return pd.DataFrame({
        'Safety Indicator': [
            'Fatal Accident Rate',
            'Hull Loss Rate', 
            'Avg Fatalities per Accident',
            'Worst Single Accident',
            'Survival Rate'
        ],
        'Value': [
            f"{(df['fatalities'] > 0).mean():.1%}",
            f"{df['cat'].eq('A1').mean():.1%}",
            f"{df['fatalities'].mean():.1f}",
            f"{df['fatalities'].max()} ({df.loc[df['fatalities'].idxmax()]['type'] if not df.empty else 'N/A'})",
            f"{1 - (df['fatalities'].sum() / df['aboard'].sum()):.1%}" if 'aboard' in df.columns else "N/A"
        ],
        'Description': [
            "Percentage of accidents with fatalities",
            "Percentage of accidents resulting in hull loss",
            "Average number of fatalities per accident",
            "Highest fatality count in single accident",
            "Percentage of passengers who survived"
        ]
    })

def create_excel_export(*dfs):
    """Create multi-sheet Excel file from dataframes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for i, df in enumerate(dfs):
            sheet_name = f"Data_{i+1}" if i > 0 else "Summary"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()