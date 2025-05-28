# streamlit_app.py
import streamlit as st
from src.trends import plot_fatalities_timeline, plot_accident_categories, plot_manufacturer_safety, plot_age_impact, plot_time_heatmap
from src.trends import get_download_section
from src.data_processing import load_data, clean_data, clean_fatalities_data, add_columns
from src.family_safety_analysis import aircraft_family_analysis_page
import base64
from pathlib import Path
from datetime import datetime


st.set_page_config(
    page_title="Aviation Accident Dashboard",
    page_icon="✈️",
    layout="wide"
)
st.title("Test App")
st.write("If this works, your basic setup is correct")

# def add_aviation_header():
#     """Fixed header with proper image positioning"""
#     try:
#         img_path = "images/sky_header.jpg"
#         img_bytes = Path(img_path).read_bytes()
#         img_base64 = base64.b64encode(img_bytes).decode()
        
#         st.markdown(
#             f"""
#             <style>
#                 .header-container {{
#                     position: relative;
#                     width: 100%;
#                     height: 280px;
#                     overflow: hidden;
#                     margin-bottom: 2rem;
#                 }}
#                 .header-image {{
#                     width: 100%;
#                     height: 100%;
#                     object-fit: cover;
#                     object-position: center 70%;  /* Adjust this to show the plane */
#                 }}
#                 .header-overlay {{
#                     position: absolute;
#                     bottom: 0;
#                     left: 0;
#                     right: 0;
#                     background: linear-gradient(
#                         to top, 
#                         rgba(0,0,0,0.7) 0%, 
#                         transparent 100%
#                     );
#                     padding: 2rem;
#                     color: white;
#                 }}
#                 .header-title {{
#                     font-size: 2.5rem;
#                     font-weight: 700;
#                     margin: 0;
#                     text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
#                 }}
#             </style>
#             <div class="header-container">
#                 <img class="header-image" src="data:image/jpeg;base64,{img_base64}" alt="Aviation header">
#                 <div class="header-overlay">
#                     <h1 class="header-title">Aviation Safety Intelligence</h1>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     except FileNotFoundError:
#         st.warning("Header image not found - using simple header")
#         st.title("Aviation Safety Intelligence")


# add_aviation_header()  # Add header first

# # App descriptions
# app_description_fr = """
# Cette application interactive permet d'explorer et d'analyser les données d'accidents aéronautiques,
# afin de mieux comprendre les tendances et améliorer la sécurité aérienne.
# """

# app_description_en = """
# This interactive application allows you to explore and analyze aviation accident data. 
# It provides dynamic visualizations to identify accident patterns
# , helping to better understand trends and enhance aviation safety.
# """

# st.markdown(app_description_fr)
# st.markdown(app_description_en)
# st.markdown("\n\n")

# try:
#     df = load_data()
#     df = clean_data(df)
#     df = add_columns(df)
# except Exception as e:
#     st.error(f"Error while loading or cleaning data: {e}")
#     st.stop()

# # Filtres avec gestion d'erreur
# try:
#     st.sidebar.header("Filter Options")
#     manufacturer_filter = st.sidebar.multiselect("Manufacturer", sorted(df['manufacturer'].dropna().unique()))
#     df_filtered_manufacturer = df[df['manufacturer'].isin(manufacturer_filter)] if manufacturer_filter else df.copy()

#     aircraft_model_options = sorted(df_filtered_manufacturer['aircraft_model'].dropna().unique())
#     aircraft_model_filter = st.sidebar.multiselect("Aircraft Model", options=aircraft_model_options)
#     df_filtered_model = df_filtered_manufacturer[df_filtered_manufacturer['aircraft_model'].isin(aircraft_model_filter)] if aircraft_model_filter else df_filtered_manufacturer.copy()

#     type_options = sorted(df_filtered_model['type'].dropna().unique())
#     type_filter = st.sidebar.multiselect("Aircraft Type", options=type_options)

#     year_filter = st.sidebar.multiselect("Year", sorted(df['year'].dropna().astype(int).unique()))
#     operator_filter = st.sidebar.multiselect("Operator", sorted(df['operator'].dropna().unique()))
#     country_filter = st.sidebar.multiselect("Country", sorted(df['country'].dropna().unique()))
#     cat_filter = st.sidebar.multiselect("Accident Category", sorted(df['cat'].dropna().unique()))

#     filtered_df = df.copy()
#     if year_filter: filtered_df = filtered_df[filtered_df['year'].isin(year_filter)]
#     if manufacturer_filter: filtered_df = filtered_df[filtered_df['manufacturer'].isin(manufacturer_filter)]
#     if aircraft_model_filter: filtered_df = filtered_df[filtered_df['aircraft_model'].isin(aircraft_model_filter)]
#     if type_filter: filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
#     if operator_filter: filtered_df = filtered_df[filtered_df['operator'].isin(operator_filter)]
#     if country_filter: filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
#     if cat_filter: filtered_df = filtered_df[filtered_df['cat'].isin(cat_filter)]
# except Exception as e:
#     st.error(f"Error while applying filters: {e}")
#     st.stop()

# # Préparation des données pour visualisation
# try:
#     df_clean = clean_fatalities_data(filtered_df)
# except Exception as e:
#     st.error(f"Error while preparing fatalities data: {e}")
#     df_clean = filtered_df.copy()

# # Tabs et visualisations
# tab_trends, tab_safety, tab_geo = st.tabs([
#     "Accident Trends & Patterns",
#     "Aircraft Family Safety",
#     "Geographic Analysis"
# ])

# with tab_trends:
#     st.header("Accident Trends & Patterns")
#     try:
#         with st.expander("Filtered Dataframe"):
#             st.dataframe(df_clean)

#         plot_fatalities_timeline(df_clean)
#         plot_accident_categories(df_clean)
#         plot_manufacturer_safety(df_clean)
#         plot_age_impact(df_clean)
#         plot_time_heatmap(df_clean)
#         get_download_section(df, df_clean)
#     except Exception as e:
#         st.error(f"Error in trend visualizations: {e}")

# with tab_safety:
#     try:
#         aircraft_family_analysis_page(filtered_df)
#     except Exception as e:
#         st.error(f"Error in aircraft family safety page: {e}")

# with tab_geo:
#     st.subheader("Accident Geographic Distribution")
#     try:
#         if {'latitude', 'longitude'}.issubset(df_clean.columns):
#             col_map, col_stats = st.columns([3, 1])
#             with col_map:
#                 st.map(df_clean[['latitude', 'longitude']].dropna(), zoom=1, use_container_width=True)
#             with col_stats:
#                 st.metric("Countries with Most Accidents", df_clean['country'].value_counts().index[0])
#                 st.metric("Highest Risk Region", df_clean['region'].value_counts().index[0])
#         else:
#             st.info("Geographic analysis not available yet")
#     except Exception as e:
#         st.error(f"Error in geographic analysis: {e}")

# def add_footer():
#     """Professional footer with GitHub CV link"""
#     current_year = datetime.now().year
    
#     st.markdown(
#         f"""
#         <style>
#             .footer {{
#                 width: 100%;
#                 padding: 1.2rem;
#                 text-align: center;
#                 background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d);
#                 margin-top: 3rem;
#                 color: white;
#                 border-radius: 8px 8px 0 0;
#             }}
#             .footer-content {{
#                 max-width: 800px;
#                 margin: 0 auto;
#                 display: flex;
#                 justify-content: space-between;
#                 align-items: center;
#             }}
#             .footer-links a {{
#                 color: white;
#                 margin: 0 12px;
#                 text-decoration: none;
#                 transition: all 0.3s ease;
#                 font-weight: 500;
#             }}
#             .footer-links a:hover {{
#                 text-decoration: underline;
#                 opacity: 0.9;
#             }}
#             .footer-name {{
#                 font-weight: 700;
#                 font-size: 1.1rem;
#             }}
#         </style>
#         <div class="footer">
#             <div class="footer-content">
#                 <div class="footer-name">© {current_year} Aviation Safety Analytics</div>
#                 <div class="footer-links">
#                     <span>Created by <strong>Rym Otsmane</strong></span>
#                     <a href="https://www.linkedin.com/in/rym-otsmane" target="_blank">
#                         <i class="fab fa-linkedin"></i> LinkedIn
#                     </a>
#                     <a href="https://github.com/grymlin/rym-otsmane-cv/blob/main/CV_Data_Rym_Otsmane.pdf" target="_blank">
#                         <i class="fas fa-file-pdf"></i> Download CV
#                     </a>
#                     <a href="https://github.com/grymlin" target="_blank">
#                         <i class="fab fa-github"></i> GitHub
#                     </a>
#                 </div>
#             </div>
#         </div>
#         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
#         """,
#         unsafe_allow_html=True
#     )
# add_footer()

