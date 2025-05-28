# family_safety_analysis.py 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt

"""
    Creation and modification:
    Creation date               : 28/05/2025

    Creation date               : 28/05/2025

    @author                     : Rym Otsmane

"""

def analyze_aircraft_families(df):
    """Enhanced aircraft family analysis with proper column creation"""
    # Data validation
    required_cols = ['type', 'fatalities', 'year', 'operator', 'cat']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing required columns for analysis")
        return None, None
    
    # Clean and prepare data
    df = df.dropna(subset=['type']).copy()
    df['manufacturer'] = df['type'].str.extract(r'^([A-Za-z]+)')[0]
    
    # Convert to numeric with error handling
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce')
    df = df.dropna(subset=['year', 'fatalities'])
    
    try:
        # Calculate statistics per aircraft type
        stats = df.groupby('type').agg({
            'fatalities': ['mean', 'sum', 'count'],
            'year': ['min', 'max'],
            'cat': lambda x: (x == 'A1').mean()  # Hull loss rate
        })
        
        # Flatten multi-index columns
        stats.columns = ['fatality_rate', 'total_fatalities', 'accident_count',
                        'first_year', 'last_year', 'hull_loss_rate']
        stats = stats.reset_index()
        
        # Merge stats back to original dataframe
        df = df.merge(stats, on='type')
        
        # Calculate service years
        df['service_years'] = df['last_year'] - df['first_year']
        
        # TF-IDF analysis for naming patterns
        tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            lowercase=False
        )
        tfidf_matrix = tfidf.fit_transform(df['type'])
        
        # Feature matrix
        features = pd.DataFrame({
            'name_similarity': cosine_similarity(tfidf_matrix).mean(axis=1),
            'fatality_rate': df['fatality_rate'],
            'accident_frequency': df['accident_count'],
            'hull_loss_rate': df['hull_loss_rate'],
            'service_years': df['service_years']
        })
        
        # Normalize features
        features = (features - features.mean()) / features.std()
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.8, min_samples=3).fit(features)
        df['family'] = clustering.labels_
        
        return df, features
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None, None

def plot_family_analysis(df, features):
    """Visualization with proper column references and actual plots"""
    if df is None or features is None:
        return
    


    # Calculate family stats using existing columns
    family_stats = df.groupby('family').agg({
        'type': ['nunique', lambda x: ', '.join(x.unique()[:3]) + '...'],
        'fatalities': ['sum', 'mean', 'count'],
        'hull_loss_rate': 'mean',
        'service_years': 'mean'
    }).reset_index()
    
    # Clean column names
    family_stats.columns = [
        'family', 'num_types', 'example_types', 
        'total_fatalities', 'mean_fatalities', 'accident_count',
        'hull_loss_rate', 'avg_service_years'
    ]
    
    # Filter out outliers
    family_stats = family_stats[family_stats['family'] != -1]
    
    if family_stats.empty:
        st.warning("No valid family clusters found")
        return
    
    # Safety score calculation
    family_stats['safety_score'] = (
        1 / (1 + family_stats['mean_fatalities']) * 
        (1 - family_stats['hull_loss_rate']) *
        np.log1p(family_stats['avg_service_years'])
    )

    # 1. Parallel Coordinates Plot
    st.subheader("Family Safety Profiles")
    fig = px.parallel_coordinates(
        family_stats,
        color='total_fatalities',
        dimensions=[
            'num_types', 'total_fatalities', 'mean_fatalities',
            'hull_loss_rate', 'avg_service_years', 'safety_score'
        ],
        labels={
            'num_types': 'Aircraft Types',
            'total_fatalities': 'Total Fatalities',
            'mean_fatalities': 'Avg Fatalities',
            'hull_loss_rate': 'Hull Loss Rate',
            'avg_service_years': 'Avg Service Years',
            'safety_score': 'Safety Score'
        },
        color_continuous_scale=px.colors.diverging.RdYlGn_r,
        width=1000, height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Safety Score Bar Chart
    st.subheader("Family Safety Scores")
    safety_chart = alt.Chart(family_stats).mark_bar().encode(
        x=alt.X('family:N', title='Family ID'),
        y=alt.Y('safety_score:Q', title='Safety Score'),
        color=alt.Color('safety_score:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
        tooltip=['family', 'example_types', 'safety_score']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(safety_chart, use_container_width=True)


def plot_family_network(df, features):
    """Network visualization of aircraft families"""
    if df is None or features is None:
        return
    
    # Prepare node data
    nodes = df.groupby(['family', 'manufacturer']).size().reset_index(name='count')
    nodes = nodes[nodes['family'] != -1]  # Remove outliers
    
    # Create edges based on feature similarity
    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0)  # Remove self-similarity
    edge_threshold = st.slider(
        "Similarity threshold for connections",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )
    
    # Create Plotly network graph
    fig = go.Figure()
    
    # Add nodes
    for _, row in nodes.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['family']],
            y=[row['count']],
            mode='markers+text',
            marker=dict(
                size=row['count']*2,
                color=row['family'],
                colorscale='Viridis'
            ),
            text=f"Family {row['family']}<br>{row['manufacturer']}",
            textposition="bottom center",
            name=f"Family {row['family']}"
        ))
    
    # Add edges (simplified for demo)
    fig.update_layout(
        title="Aircraft Family Network",
        xaxis_title="Family Cluster",
        yaxis_title="Number of Types",
        showlegend=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_type_breakdown(family_data):
    """Visualization of aircraft types in family"""
    type_stats = family_data.groupby('type').agg({
        'fatalities': ['sum', 'mean', 'count']
    }).reset_index()
    type_stats.columns = ['type', 'total_fatalities', 'mean_fatalities', 'accident_count']
    
    chart = alt.Chart(type_stats).mark_bar().encode(
        x=alt.X('type:N', sort='-y', title='Aircraft Type'),
        y=alt.Y('total_fatalities:Q', title='Total Fatalities'),
        color=alt.Color('mean_fatalities:Q', scale=alt.Scale(scheme='reds')),
        tooltip=['type', 'total_fatalities', 'mean_fatalities', 'accident_count']
    ).properties(
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)

def aircraft_family_analysis_page(df):
    """Complete analysis page with interactive controls"""
    st.header("Aircraft Family Safety Analysis")
    st.markdown("""
    Identifies groups of aircraft with similar naming patterns **and** accident characteristics,
    helping detect potential design or maintenance issues.
    """)
    if len(df) > 10000:  # Adjust threshold based on your expected size
        raise ValueError("Dataset too large for cloud processing")
    
    # Explicitly reduce memory usage
    df = df.copy()
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['integer']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    with st.expander("Analysis Methodology"):
        st.markdown("""
        1. **Naming Patterns**: Character-level TF-IDF analysis of aircraft type names  
        2. **Accident Stats**: Fatality rates, hull loss rates, service years  
        3. **Clustering**: DBSCAN algorithm groups similar aircraft  
        4. **Safety Scoring**: Combines multiple safety metrics  
        """)
    
    with st.spinner("Analyzing aircraft families (this may take a moment)..."):
        result, features = analyze_aircraft_families(df)
    
    if result is not None:
        st.success(f"Analyzed {len(result)} accidents across {result['type'].nunique()} aircraft types")
        plot_family_analysis(result, features)
    
    # Initialize session state for family selection
    if 'selected_family' not in st.session_state:
        st.session_state.selected_family = None
    

    if st.button("Analyze aircraft Families", type="primary",help="An aircraft family clusters models with shared design (e.g., Boeing 737)."
        " DBSCAN groups them by names, accident rates, and service years to spot systemic risks."):
        with st.spinner("Analyzing aircraft families..."):
            result, features = analyze_aircraft_families(df)
            
            if result is not None:
                st.session_state.analysis_result = result
                st.session_state.analysis_features = features
                st.success(f"Analyzed {len(result)} accidents across {result['type'].nunique()} aircraft types")
    
    # Check if we have analysis results to show
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        features = st.session_state.analysis_features
        
        # Family selection - uses session state to maintain selection
        st.subheader("Family Selection")
        family_options = sorted(result['family'].unique())
        selected_family = st.selectbox(
            "Select family to inspect",
            options=family_options,
            index=family_options.index(st.session_state.selected_family) if st.session_state.selected_family in family_options else 0,
            key='family_selectbox'
        )
        
        # Update session state when selection changes
        if selected_family != st.session_state.selected_family:
            st.session_state.selected_family = selected_family
            st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
        
        # Show visualizations for selected family
        if st.session_state.selected_family is not None:
            family_data = result[result['family'] == st.session_state.selected_family]
            
            # Family overview metrics
            st.subheader(f"Family {st.session_state.selected_family} Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                types_count = family_data['type'].nunique()
                st.metric("Aircraft Types", types_count)
            with col2:
                total_fatal = family_data['fatalities'].sum()
                st.metric("Total Fatalities", int(total_fatal))
            with col3:
                avg_fatal = family_data['fatalities'].mean()
                st.metric("Avg Fatalities", f"{avg_fatal:.1f}")
            
            # Show example aircraft
            example_types = ", ".join(family_data['type'].unique()[:3])
            st.caption(f"Example aircraft: {example_types}")
            
            # Safety metrics visualization
            st.subheader("Safety Metrics")
            plot_family_safety_metrics(family_data)
            
            # Aircraft type breakdown
            st.subheader("Aircraft Type Breakdown")
            plot_type_breakdown(family_data)
            
            # Raw data view
            with st.expander("View Detailed Family Data"):
                st.dataframe(
                    family_data[
                        ['type', 'manufacturer', 'fatalities', 'year', 
                         'operator', 'cat', 'fatality_rate', 'hull_loss_rate']
                    ].sort_values('fatalities', ascending=False),
                    height=300
                )
            
            # Download option
            st.download_button(
                "Download Family Data",
                data=family_data.to_csv(index=False),
                file_name=f"aircraft_family_{st.session_state.selected_family}.csv"
            )

def plot_family_safety_metrics(family_data):
    """Visualization of safety metrics for selected family"""
    metrics = {
        'Fatality Rate': family_data['fatality_rate'].mean(),
        'Hull Loss Rate': family_data['hull_loss_rate'].mean(),
        'Service Years': family_data['service_years'].mean(),
        'Accident Count': family_data['accident_count'].mean()
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=['#EF553B', '#00CC96', '#636EFA', '#AB63FA']
    ))
    
    fig.update_layout(
        yaxis_title="Metric Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

