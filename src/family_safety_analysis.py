# aircraft_family_analysis.py - Tailored for Historical Aircraft Data
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import re
from datetime import datetime

"""
Aircraft Family Analysis optimized for historical aircraft data
with better pattern recognition for vintage and classic aircraft
"""

def extract_aircraft_features(df):
    """Extract meaningful aircraft characteristics for family grouping"""
    df = df.copy()
    
    # 1. Enhanced manufacturer extraction for historical aircraft
    def extract_manufacturer(aircraft_type):
        aircraft_type = str(aircraft_type).upper()
        
        # Historical manufacturers mapping
        manufacturer_patterns = {
            'DOUGLAS': ['DOUGLAS', 'DC-', 'DST-', 'C-47', 'C-53', 'C-54', 'R4D', 'R5D'],
            'BOEING': ['BOEING', 'B-', '707-', '727-', '737-', '747-', '757-', '767-', '777-', '787-'],
            'LOCKHEED': ['LOCKHEED', 'L-', 'C-60', 'R5O', 'P2V', 'CONSTELLATION', 'ELECTRA', 'LODESTAR', 'HUDSON'],
            'JUNKERS': ['JUNKERS', 'JU-', 'JUG-', 'G.24', 'G.31', 'G.38'],
            'FORD': ['FORD', 'TRI-MOTOR'],
            'CURTISS': ['CURTISS', 'CONDOR'],
            'FOKKER': ['FOKKER', 'F.'],
            'HANDLEY PAGE': ['HANDLEY PAGE', 'HP.', 'W.8', 'W.9', 'W.10', 'O/'],
            'CONSOLIDATED': ['CONSOLIDATED', 'PBY', 'CATALINA', 'CANSO', 'PB2B'],
            'ANTONOV': ['ANTONOV', 'AN-'],
            'ILYUSHIN': ['ILYUSHIN', 'IL-'],
            'TUPOLEV': ['TUPOLEV', 'ANT-'],
            'SAVOIA-MARCHETTI': ['SAVOIA-MARCHETTI', 'SM-', 'S-66', 'S.73'],
            'SHORT': ['SHORT', 'S.8', 'S.17', 'S.23', 'S.25', 'S.26', 'S.30', 'SUNDERLAND', 'EMPIRE'],
            'SIKORSKY': ['SIKORSKY', 'S-41', 'S-42', 'S-43', 'VS-44', 'Y1OA'],
            'VICKERS': ['VICKERS', 'VIKING', 'VISCOUNT', 'VALENTIA', 'VALETTA'],
            'AVRO': ['AVRO', 'LANCASTRIAN', 'TUDOR', 'JETLINER'],
            'BRISTOL': ['BRISTOL', 'FREIGHTER'],
            'CONVAIR': ['CONVAIR', 'CV-'],
            'MARTIN': ['MARTIN', 'M-130', 'PBM', '2-0-2'],
            'FAIRCHILD': ['FAIRCHILD', 'C-82', 'C-119', 'SA227'],
            'GRUMMAN': ['GRUMMAN', 'G-159', 'G-73'],
            'EMBRAER': ['EMBRAER', 'EMB-'],
            'BEECH': ['BEECH', 'KING AIR'],
            'LEARJET': ['LEARJET'],
            'HAWKER SIDDELEY': ['HAWKER SIDDELEY', 'HS-'],
            'CASA': ['CASA', 'C-207'],
            'DORNIER': ['DORNIER', 'DO'],
            'LET': ['LET', 'L-410'],
            'AIRBUS': ['AIRBUS', 'A300', 'A310', 'A320', 'A330', 'A340', 'A350', 'A380']
        }
        
        for manufacturer, patterns in manufacturer_patterns.items():
            if any(pattern in aircraft_type for pattern in patterns):
                return manufacturer
        
        # Fallback: extract first word
        first_word = aircraft_type.split()[0] if ' ' in aircraft_type else aircraft_type
        return first_word[:20]  # Limit length
    
    df['manufacturer'] = df['type'].apply(extract_manufacturer)
    
    # 2. Enhanced model extraction
    def extract_model_family(aircraft_type):
        aircraft_type = str(aircraft_type).upper()
        
        # DC family
        if 'DC-' in aircraft_type:
            model_match = re.search(r'DC-(\d+)', aircraft_type)
            return f"DC-{model_match.group(1)}" if model_match else 'DC-UNKNOWN'
        
        # Boeing commercial jets
        elif any(x in aircraft_type for x in ['707', '727', '737', '747', '757', '767', '777', '787']):
            model_match = re.search(r'(7\d7)', aircraft_type)
            return model_match.group(1) if model_match else 'BOEING-JET'
        
        # Junkers
        elif 'JU-52' in aircraft_type or 'JU 52' in aircraft_type:
            return 'JU-52'
        elif 'G.24' in aircraft_type:
            return 'G.24'
        elif 'G.31' in aircraft_type:
            return 'G.31'
        
        # Lockheed Constellation family
        elif 'CONSTELLATION' in aircraft_type or 'L-749' in aircraft_type or 'L-649' in aircraft_type:
            return 'CONSTELLATION'
        elif 'LODESTAR' in aircraft_type or 'L-18' in aircraft_type:
            return 'LODESTAR'
        elif 'ELECTRA' in aircraft_type or 'L-14' in aircraft_type:
            return 'ELECTRA'
        
        # Ford Tri-Motor family
        elif 'TRI-MOTOR' in aircraft_type or 'FORD' in aircraft_type:
            return 'TRI-MOTOR'
        
        # Catalina family
        elif any(x in aircraft_type for x in ['PBY', 'CATALINA', 'CANSO']):
            return 'CATALINA'
        
        # Sunderland family
        elif 'SUNDERLAND' in aircraft_type:
            return 'SUNDERLAND'
        
        # Antonov family
        elif 'AN-' in aircraft_type:
            model_match = re.search(r'AN-(\d+)', aircraft_type)
            return f"AN-{model_match.group(1)}" if model_match else 'AN-UNKNOWN'
        
        # Ilyushin family
        elif 'IL-' in aircraft_type:
            model_match = re.search(r'IL-(\d+)', aircraft_type)
            return f"IL-{model_match.group(1)}" if model_match else 'IL-UNKNOWN'
        
        # Default: try to extract model number
        else:
            model_match = re.search(r'(\d+)', aircraft_type)
            return model_match.group(1) if model_match else 'UNKNOWN'
    
    df['model_family'] = df['type'].apply(extract_model_family)
    
    # 3. Enhanced aircraft categorization based on historical context
    def categorize_aircraft(aircraft_type):
        aircraft_type = str(aircraft_type).upper()
        
        # Large transport/airliner (4+ engines or large capacity)
        if any(pattern in aircraft_type for pattern in [
            'DC-6', 'DC-7', 'DC-8', 'DC-10', 'L-1049', 'L-1649', 'CONSTELLATION',
            '707', '720', '747', '767', '777', '787', '880', '990',
            'COMET', 'CARAVELLE', 'TRIDENT', 'VC10', 'IL-62', 'TU-104', 'TU-114', 'TU-134',
            'CONVAIR 880', 'CONVAIR 990', 'SUD CARAVELLE'
        ]):
            return 'large_transport'
        
        # Medium transport (twin-engine airliners)
        elif any(pattern in aircraft_type for pattern in [
            'DC-3', 'DC-4', 'DC-5', 'C-47', 'C-54', 'MARTIN 2-0-2', 'MARTIN 4-0-4',
            'CONVAIR 240', 'CONVAIR 340', 'CONVAIR 440', 'CV-', 'VISCOUNT', 'VANGUARD',
            'IL-14', 'IL-18', 'TU-124', 'FRIENDSHIP', 'HERALD', 'YS-11',
            '727', '737', 'CARAVELLE', 'BAC 111', 'DC-9', 'F28', 'TRIDENT'
        ]):
            return 'medium_transport'
        
        # Small transport/regional
        elif any(pattern in aircraft_type for pattern in [
            'TWIN OTTER', 'BRITTEN-NORMAN', 'BN-2', 'ISLANDER', 'TRISLANDER',
            'BEECH 18', 'BEECH 99', 'PIPER NAVAJO', 'CESSNA 402', 'CESSNA 404',
            'EMB-110', 'BANDEIRANTE', 'METRO', 'L-410', 'AN-24', 'AN-26', 'AN-2',
            'DHC-', 'OTTER', 'BEAVER', 'HERALD'
        ]):
            return 'small_transport'
        
        # Flying boats
        elif any(pattern in aircraft_type for pattern in [
            'SUNDERLAND', 'CATALINA', 'PBY', 'CANSO', 'EMPIRE', 'SOLENT', 'PRINCESS',
            'MARTIN M-130', 'BOEING 314', 'S-42', 'S-43', 'SANDRINGHAM'
        ]):
            return 'flying_boat'
        
        # Early airliners (pre-1940)
        elif any(pattern in aircraft_type for pattern in [
            'TRI-MOTOR', 'FORD', 'FOKKER F.', 'JUNKERS G.', 'HANDLEY PAGE',
            'FARMAN', 'GOLIATH', 'HERCULES', 'ARGOSY', 'ATALANTA', 'SCYLLA',
            'CONDOR', 'COMMODORE'
        ]):
            return 'early_airliner'
        
        # Cargo/freight
        elif any(pattern in aircraft_type for pattern in [
            'FREIGHTER', 'CARGO', 'C-130', 'C-119', 'C-82', 'HERCULES',
            'BOXCAR', 'PACKET', 'IL-76', 'AN-12', 'TRANSALL'
        ]):
            return 'cargo'
        
        # Business/executive
        elif any(pattern in aircraft_type for pattern in [
            'LEARJET', 'GULFSTREAM', 'FALCON', 'CITATION', 'HAWKER', 'SABRELINER',
            'JET COMMANDER', 'KING AIR', 'QUEEN AIR', 'AERO COMMANDER'
        ]):
            return 'business'
        
        # Default based on era and size hints
        else:
            return 'general_aviation'
    
    df['aircraft_category'] = df['type'].apply(categorize_aircraft)
    
    # 4. Technology era classification (more granular for historical aircraft)
    def get_aircraft_era(year):
        if pd.isna(year):
            return 'unknown'
        elif year < 1930:
            return 'pioneer_era'  # 1903-1930
        elif year < 1940:
            return 'golden_age'   # 1930-1940
        elif year < 1950:
            return 'war_era'      # 1940-1950
        elif year < 1960:
            return 'early_jets'   # 1950-1960
        elif year < 1970:
            return 'jet_age'      # 1960-1970
        elif year < 1980:
            return 'wide_body'    # 1970-1980
        elif year < 1990:
            return 'modern_jets'  # 1980-1990
        elif year < 2000:
            return 'advanced'     # 1990-2000
        else:
            return 'next_gen'     # 2000+
    
    df['era'] = df['year'].apply(get_aircraft_era)
    
    # 5. Operational characteristics from accident patterns
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Calculate safety metrics per aircraft type
    operational_features = df.groupby('type').agg({
        'fatalities': ['mean', 'std', 'sum', 'count'],
        'year': ['min', 'max', 'count'],
        'cat': lambda x: (x.str.upper() == 'A1').sum() / len(x) if len(x) > 0 else 0,
    }).reset_index()
    
    # Flatten column names
    operational_features.columns = [
        'type', 'avg_fatalities', 'fatality_std', 'total_fatalities', 'accident_count',
        'first_accident', 'last_accident', 'total_years', 'hull_loss_rate'
    ]
    
    # Merge back with main dataframe
    df = df.merge(operational_features, on='type', how='left')
    
    return df

def create_engineering_families(df):
    """Create aircraft families using engineering-based clustering"""
    
    # Extract enhanced features
    df_enhanced = extract_aircraft_features(df)
    
    # Create feature matrix for clustering
    categorical_features = ['manufacturer', 'model_family', 'aircraft_category', 'era']
    numerical_features = ['avg_fatalities', 'accident_count', 'hull_loss_rate', 'year']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_enhanced[categorical_features], prefix=categorical_features)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(df_enhanced[numerical_features].fillna(0))
    numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features, index=df_enhanced.index)
    
    # Combine features
    feature_matrix = pd.concat([df_encoded, numerical_df], axis=1)
    
    # Apply clustering methods
    
    # 1. DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    df_enhanced['dbscan_cluster'] = dbscan.fit_predict(feature_matrix)
    
    # 2. K-Means clustering
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    df_enhanced['kmeans_cluster'] = kmeans.fit_predict(feature_matrix)
    
    # 3. Hybrid logical grouping
    def create_logical_family(row):
        return f"{row['manufacturer']}_{row['aircraft_category']}_{row['era']}"
    
    df_enhanced['logical_family'] = df_enhanced.apply(create_logical_family, axis=1)
    
    # 4. Model-based family grouping
    def create_model_family(row):
        return f"{row['manufacturer']}_{row['model_family']}"
    
    df_enhanced['model_based_family'] = df_enhanced.apply(create_model_family, axis=1)
    
    return df_enhanced, feature_matrix, scaler

def analyze_family_performance(df_families):
    """Analyze performance metrics for each aircraft family"""
    
    family_methods = ['dbscan_cluster', 'kmeans_cluster', 'logical_family', 'model_based_family']
    analysis_results = {}
    
    for method in family_methods:
        # Skip DBSCAN noise points (-1)
        if method == 'dbscan_cluster':
            df_method = df_families[df_families[method] != -1].copy()
        else:
            df_method = df_families.copy()
        
        # Calculate family statistics
        family_stats = df_method.groupby(method).agg({
            'fatalities': ['sum', 'mean', 'std', 'count'],
            'accident_count': 'first',  # This is already aggregated per type
            'hull_loss_rate': 'mean',
            'year': ['min', 'max'],
            'type': 'nunique'
        }).round(2)
        
        # Flatten column names
        family_stats.columns = [
            'total_fatalities', 'avg_fatalities_per_accident', 'fatality_std',
            'total_accidents', 'avg_accident_count', 'avg_hull_loss_rate',
            'first_year', 'last_year', 'aircraft_types_count'
        ]
        
        # Calculate safety score (lower is better)
        family_stats['safety_score'] = (
            family_stats['avg_fatalities_per_accident'] * 0.4 +
            family_stats['avg_hull_loss_rate'] * 0.3 +
            (family_stats['total_accidents'] / family_stats['aircraft_types_count']) * 0.3
        )
        
        # Add operational span
        family_stats['operational_span'] = family_stats['last_year'] - family_stats['first_year']
        
        analysis_results[method] = family_stats.sort_values('safety_score')
    
    return analysis_results

def create_family_visualizations(df_families, analysis_results):
    """Create comprehensive visualizations for aircraft families"""
    
    visualizations = {}
    
    # 1. Family Size Distribution
    fig_sizes = go.Figure()
    
    for i, (method, stats) in enumerate(analysis_results.items()):
        fig_sizes.add_trace(go.Histogram(
            x=stats['aircraft_types_count'],
            name=method.replace('_', ' ').title(),
            opacity=0.7,
            nbinsx=20
        ))
    
    fig_sizes.update_layout(
        title="Aircraft Family Size Distribution by Clustering Method",
        xaxis_title="Number of Aircraft Types per Family",
        yaxis_title="Number of Families",
        barmode='overlay'
    )
    visualizations['family_sizes'] = fig_sizes
    
    # 2. Safety Score Comparison
    fig_safety = go.Figure()
    
    for method, stats in analysis_results.items():
        top_families = stats.head(15)  # Top 15 safest families
        
        fig_safety.add_trace(go.Bar(
            name=method.replace('_', ' ').title(),
            x=top_families.index,
            y=top_families['safety_score'],
            text=top_families['aircraft_types_count'],
            texttemplate='%{text} types',
            textposition='outside'
        ))
    
    fig_safety.update_layout(
        title="Top 15 Safest Aircraft Families by Method",
        xaxis_title="Aircraft Family",
        yaxis_title="Safety Score (Lower = Safer)",
        barmode='group',
        xaxis={'tickangle': 45}
    )
    visualizations['safety_comparison'] = fig_safety
    
    # 3. Era vs Category Heatmap
    era_category_matrix = pd.crosstab(df_families['era'], df_families['aircraft_category'])
    
    fig_heatmap = px.imshow(
        era_category_matrix.values,
        x=era_category_matrix.columns,
        y=era_category_matrix.index,
        title="Aircraft Types by Era and Category",
        labels=dict(x="Aircraft Category", y="Technology Era", color="Count"),
        aspect="auto"
    )
    visualizations['era_category_heatmap'] = fig_heatmap
    
    # 4. Manufacturer Timeline
    manufacturer_timeline = df_families.groupby(['manufacturer', 'year']).size().reset_index(name='accidents')
    
    fig_timeline = px.scatter(
        manufacturer_timeline[manufacturer_timeline['manufacturer'].isin(
            manufacturer_timeline['manufacturer'].value_counts().head(10).index
        )],
        x='year',
        y='manufacturer',
        size='accidents',
        title="Major Aircraft Manufacturers Timeline (Top 10 by Accident Count)",
        labels={'accidents': 'Number of Accidents'}
    )
    visualizations['manufacturer_timeline'] = fig_timeline
    
    # 5. Family Performance Scatter
    best_method_stats = analysis_results['logical_family']  # Use logical family as reference
    
    fig_scatter = px.scatter(
        best_method_stats.reset_index(),
        x='avg_fatalities_per_accident',
        y='avg_hull_loss_rate',
        size='aircraft_types_count',
        hover_name='logical_family',
        title="Aircraft Family Performance Analysis",
        labels={
            'avg_fatalities_per_accident': 'Average Fatalities per Accident',
            'avg_hull_loss_rate': 'Hull Loss Rate',
            'aircraft_types_count': 'Family Size'
        }
    )
    visualizations['performance_scatter'] = fig_scatter
    
    return visualizations

def aircraft_family_analysis_page(df):
    """Main analysis page with improved family detection"""
    
    st.header(" Aircraft Family Safety Analysis")
    st.markdown("""
    This analysis groups aircraft based on engineering characteristics:
    - **Manufacturer & Model Patterns** (Douglas DC-3 family, Boeing 707 family, etc.)
    - **Aircraft Category** (large transport, medium transport, regional, flying boats, etc.)
    - **Technology Era** (pioneer era, golden age, jet age, modern jets, etc.)
    - **Operational Safety Patterns** (accident rates, fatality patterns, hull loss rates)
    """)
    
    # Data size check and info
    st.write(f"**Dataset Info:** {len(df):,} accidents involving {df['type'].nunique():,} different aircraft types")
    
    if len(df) > 50000:
        st.warning("Large dataset detected. Consider applying filters for better performance.")
        
    # Show some sample data
    with st.expander("Sample Data Preview"):
        st.write("**Aircraft Type Distribution (Top 15):**")
        top_types = df['type'].value_counts().head(15)
        fig = px.bar(
            x=top_types.values,
            y=top_types.index,
            orientation='h',
            title="Most Common Aircraft Types in Dataset"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Analysis Methodology"):
        st.markdown("""
        ### Feature Engineering:
        1. ** Manufacturer Detection**: Recognizes historical manufacturers (Douglas, Lockheed, Junkers, etc.)
        2. **Model Family Grouping**: Groups variants (DC-3/C-47, Boeing 707 series, etc.)
        3. **Aircraft Categorization**: 
           - Large Transport (4+ engines, intercontinental)
           - Medium Transport (twin-engine airliners)
           - Small Transport/Regional
           - Flying Boats (Catalina, Sunderland, etc.)
           - Early Airliners (Ford Tri-Motor, Junkers G.24, etc.)
           - Cargo/Freight, Business Jets
        4. **Technology Eras**: Pioneer (pre-1930), Golden Age (1930-40), War Era (1940-50), 
           Early Jets (1950-60), Jet Age (1960-70), Wide-body (1970-80), Modern (1980-90), 
           Advanced (1990-2000), Next-gen (2000+)
        5. **Safety Metrics**: Fatality rates, hull loss rates, accident frequency patterns
        
        ### Clustering Methods:
        - **DBSCAN**: Finds natural density-based clusters of similar aircraft
        - **K-Means**: Creates balanced centroid-based groups  
        - **Hybrid**: Logical grouping by manufacturer + category + era
        - **Model-based**: Groups by manufacturer + specific model family
        """)
    
    if st.button("Analyze Aircraft Families", type="primary"):
        with st.spinner("Analyzing aircraft families using engineering characteristics..."):
            
            try:
                # Perform analysis
                df_families, feature_matrix, features_scaled = create_engineering_families(df)
                family_analysis = analyze_family_performance(df_families)
                visualizations = create_family_visualizations(df_families, family_analysis)
                
                st.success("✅ Aircraft family analysis completed!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Family Overview", "Top Performers", "Performance Analysis", 
                    "Historical Trends", "Detailed Statistics"
                ])
                
                with tab1:
                    st.subheader("Aircraft Family Classification Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(visualizations['family_sizes'], use_container_width=True)
                    with col2:
                        st.plotly_chart(visualizations['era_category_heatmap'], use_container_width=True)
                    
                    st.markdown("### Family Method Comparison")
                    method_comparison = pd.DataFrame({
                        'Method': ['DBSCAN Clustering', 'K-Means Clustering', 'Logical Grouping', 'Model-Based'],
                        'Families Created': [
                            len(family_analysis['dbscan_cluster']),
                            len(family_analysis['kmeans_cluster']),
                            len(family_analysis['logical_family']),
                            len(family_analysis['model_based_family'])
                        ],
                        'Avg Family Size': [
                            family_analysis['dbscan_cluster']['aircraft_types_count'].mean(),
                            family_analysis['kmeans_cluster']['aircraft_types_count'].mean(),
                            family_analysis['logical_family']['aircraft_types_count'].mean(),
                            family_analysis['model_based_family']['aircraft_types_count'].mean()
                        ]
                    })
                    st.dataframe(method_comparison.round(2))
                
                with tab2:
                    st.subheader("🏆 Safest Aircraft Families")
                    st.plotly_chart(visualizations['safety_comparison'], use_container_width=True)
                    
                    # Show top 10 safest families from logical grouping
                    st.markdown("### Top 10 Safest Aircraft Families (Logical Grouping)")
                    top_safe = family_analysis['logical_family'].head(10)[
                        ['total_fatalities', 'avg_fatalities_per_accident', 'total_accidents', 
                         'aircraft_types_count', 'safety_score', 'operational_span']
                    ]
                    st.dataframe(top_safe)
                
                with tab3:
                    st.subheader("📈 Family Performance Analysis")
                    st.plotly_chart(visualizations['performance_scatter'], use_container_width=True)
                    
                    # Performance insights
                    st.markdown("### Key Performance Insights")
                    logical_stats = family_analysis['logical_family']
                    
                    safest_family = logical_stats.index[0]
                    largest_family = logical_stats.loc[logical_stats['aircraft_types_count'].idxmax()].name
                    longest_service = logical_stats.loc[logical_stats['operational_span'].idxmax()].name
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Safest Family", safest_family, 
                                f"Safety Score: {logical_stats.loc[safest_family, 'safety_score']:.2f}")
                    with col2:
                        st.metric("Largest Family", largest_family,
                                f"{logical_stats.loc[largest_family, 'aircraft_types_count']} aircraft types")
                    with col3:
                        st.metric("Longest Service", longest_service,
                                f"{logical_stats.loc[longest_service, 'operational_span']} years")
                
                with tab4:
                    st.subheader("🕒 Historical Development Trends")
                    st.plotly_chart(visualizations['manufacturer_timeline'], use_container_width=True)
                    
                    # Era analysis
                    st.markdown("### Aircraft Development by Era")
                    era_stats = df_families.groupby('era').agg({
                        'type': 'nunique',
                        'fatalities': 'sum',
                        'accident_count': 'sum'
                    }).rename(columns={
                        'type': 'Unique Aircraft Types',
                        'fatalities': 'Total Fatalities',
                        'accident_count': 'Total Accidents'
                    })
                    st.dataframe(era_stats)
                
                with tab5:
                    st.subheader("🔍 Detailed Family Statistics")
                    
                    # Method selector
                    method_options = {
                        'Logical Grouping': 'logical_family',
                        'Model-Based': 'model_based_family',
                        'K-Means Clustering': 'kmeans_cluster',
                        'DBSCAN Clustering': 'dbscan_cluster'
                    }
                    
                    selected_method = st.selectbox("Select Grouping Method:", options=list(method_options.keys()))
                    method_key = method_options[selected_method]
                    
                    # Show detailed statistics
                    detailed_stats = family_analysis[method_key]
                    st.dataframe(detailed_stats)
                    
                    # Download option
                    csv = detailed_stats.to_csv()
                    st.download_button(
                        label="📥 Download Family Statistics as CSV",
                        data=csv,
                        file_name=f"aircraft_family_stats_{method_key}.csv",
                        mime="text/csv"
                    )
                
                # Additional insights section
                st.markdown("---")
                st.subheader("Key Findings")
                
                with st.expander("Analysis Summary"):
                    logical_families = family_analysis['logical_family']
                    
                    st.markdown(f"""
                    ### Aircraft Family Analysis Results
                    
                    **Dataset Overview:**
                    - **Total Aircraft Types Analyzed:** {df['type'].nunique():,}
                    - **Total Accidents:** {len(df):,}
                    - **Families Identified:** {len(logical_families)} (using logical grouping)
                    - **Analysis Period:** {df['year'].min():.0f} - {df['year'].max():.0f}
                    
                    **Top Safety Performers:**
                    - **Safest Family:** {logical_families.index[0]} (Safety Score: {logical_families.iloc[0]['safety_score']:.2f})
                    - **Largest Safe Family:** {logical_families[logical_families['aircraft_types_count'] >= 3].index[0]}
                    - **Most Experienced:** {logical_families.loc[logical_families['operational_span'].idxmax()].name} ({logical_families['operational_span'].max():.0f} years of service)
                    
                    **Historical Insights:**
                    - **Pioneer Era (pre-1930):** {len(df_families[df_families['era'] == 'pioneer_era'])} accidents
                    - **Golden Age (1930-1940):** {len(df_families[df_families['era'] == 'golden_age'])} accidents
                    - **Jet Age Impact:** Significant safety improvements visible from 1960s onward
                    - **Modern Aviation:** Latest aircraft families show improved safety metrics
                    
                    **Manufacturer Patterns:**
                    - **Most Represented:** {df_families['manufacturer'].value_counts().index[0]} ({df_families['manufacturer'].value_counts().iloc[0]} accidents)
                    - **Historical Leaders:** Douglas, Boeing, and Lockheed dominated different eras
                    - **Emerging Players:** Regional manufacturers gained prominence in later periods
                    """)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Try filtering the dataset or check data quality")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.write("**Dataset columns:**", list(df.columns))
                    st.write("**Data types:**", df.dtypes)
                    st.write("**Sample data:**")
                    st.dataframe(df.head())

def create_aircraft_similarity_analysis(df_families):
    """Create similarity analysis between aircraft types"""
    
    # Calculate similarity based on operational characteristics
    aircraft_profiles = df_families.groupby('type').agg({
        'manufacturer': 'first',
        'model_family': 'first',
        'aircraft_category': 'first',
        'era': 'first',
        'avg_fatalities': 'first',
        'accident_count': 'first',
        'hull_loss_rate': 'first',
        'year': 'mean'
    }).reset_index()
    
    return aircraft_profiles

def export_family_data(df_families, analysis_results):
    """Export family analysis results for external use"""
    
    export_data = {}
    
    # Family classifications
    family_classifications = df_families[['type', 'manufacturer', 'model_family', 
                                       'aircraft_category', 'era', 'logical_family', 
                                       'model_based_family']].drop_duplicates()
    export_data['classifications'] = family_classifications
    
    # Family performance metrics
    for method, stats in analysis_results.items():
        export_data[f'performance_{method}'] = stats
    
    # Summary statistics
    summary_stats = {
        'total_aircraft_types': df_families['type'].nunique(),
        'total_accidents': len(df_families),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods_used': list(analysis_results.keys())
    }
    export_data['summary'] = pd.DataFrame([summary_stats])
    
    return export_data

# Additional utility functions for enhanced analysis

def identify_aircraft_anomalies(df_families):
    """Identify aircraft types with unusual safety patterns"""
    
    # Calculate z-scores for key metrics
    metrics = ['avg_fatalities', 'accident_count', 'hull_loss_rate']
    
    for metric in metrics:
        mean_val = df_families[metric].mean()
        std_val = df_families[metric].std()
        df_families[f'{metric}_zscore'] = (df_families[metric] - mean_val) / std_val
    
    # Identify outliers (z-score > 2 or < -2)
    anomalies = df_families[
        (abs(df_families['avg_fatalities_zscore']) > 2) |
        (abs(df_families['accident_count_zscore']) > 2) |
        (abs(df_families['hull_loss_rate_zscore']) > 2)
    ][['type', 'manufacturer', 'aircraft_category'] + 
      [col for col in df_families.columns if 'zscore' in col]]
    
    return anomalies

def create_safety_evolution_timeline(df_families):
    """Create timeline showing safety evolution by era and category"""
    
    safety_timeline = df_families.groupby(['era', 'aircraft_category']).agg({
        'avg_fatalities': 'mean',
        'hull_loss_rate': 'mean',
        'accident_count': 'sum',
        'type': 'nunique'
    }).reset_index()
    
    # Add era ordering for proper timeline
    era_order = ['pioneer_era', 'golden_age', 'war_era', 'early_jets', 
                 'jet_age', 'wide_body', 'modern_jets', 'advanced', 'next_gen']
    
    safety_timeline['era_num'] = safety_timeline['era'].map(
        {era: i for i, era in enumerate(era_order)}
    )
    
    return safety_timeline.sort_values('era_num')

def generate_family_recommendations(analysis_results):
    """Generate recommendations based on family analysis"""
    
    logical_families = analysis_results['logical_family']
    
    recommendations = []
    
    # Safest families recommendation
    top_safe = logical_families.head(5)
    recommendations.append({
        'category': 'Safest Aircraft Families',
        'description': 'Families with the lowest safety scores (best safety record)',
        'families': list(top_safe.index),
        'metric': 'Safety Score',
        'values': top_safe['safety_score'].tolist()
    })
    
    # Most experienced families
    experienced = logical_families.nlargest(5, 'operational_span')
    recommendations.append({
        'category': 'Most Experienced Families',
        'description': 'Families with longest operational history',
        'families': list(experienced.index),
        'metric': 'Years of Service',
        'values': experienced['operational_span'].tolist()
    })
    
    # Diverse families (many aircraft types)
    diverse = logical_families.nlargest(5, 'aircraft_types_count')
    recommendations.append({
        'category': 'Most Diverse Families',
        'description': 'Families with most aircraft type variants',
        'families': list(diverse.index),
        'metric': 'Aircraft Types',
        'values': diverse['aircraft_types_count'].tolist()
    })
    
    return recommendations





