# data_processing.py
import pandas as pd

"""
    Creation and modification:
    Creation date               : 28/05/2025

    Creation date               : 28/05/2025

    @author                     : Rym Otsmane

"""
data_path = 'data/aviation-accident.csv'

def load_data():
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean date field
    df['year'] = pd.to_numeric(df['year'], errors='coerce')  # Convert year to numeric, NaN if unknown
    df['date'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')  # Convert year to datetime (Jan 1)

    # Clean year field
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    return df

def clean_fatalities_data(df):
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered['fatalities'].notna()]
    df_filtered = df_filtered[df_filtered['country'].notna()]
    df_filtered = df_filtered[df_filtered['type'].notna()]

    df_filtered['fatalities'] = pd.to_numeric(df_filtered['fatalities'], errors='coerce')
    df_filtered = df_filtered[df_filtered['fatalities'].notna()]

    return df_filtered

def add_columns(df):
    type_clean = df['type'].fillna('')

    df['manufacturer'] = type_clean.str.extract(r'^(\w+)', expand=False)
    df['aircraft_model'] = type_clean.str.extract(r'^(.*?\d{1,3})', expand=False)

    df = df[[
    'date', 'year', 'type', 'manufacturer', 'aircraft_model',
    'registration', 'operator', 'fatalities',
    'location', 'country', 'cat'
]]
    df['manufacturer'] = df['manufacturer'].replace({'A':'A.W', 'B':'B.W'})
    df['manufacturer'] = df['manufacturer'].str.replace('AÃ', 'Aérospatiale')
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.replace('AÃ©rospatiale', 'Aérospatiale')
    return df
def clean_operator(df):
    """
    Cleans the 'operator' column:
    - Removes ', op.for ...' and everything after it.
    - Fixes common encoding issues.
    - Strips whitespace.
    - Sets empty, whitespace-only, or 'unknown' (case-insensitive) to 'Unknown'.
    """
    df = df.copy()
    # Remove ', op.for ...'
    df['operator'] = df['operator'].str.replace(r',\s*op\.for.*$', '', regex=True)
    # Fix common encoding issues
    replacements = {
        'AerotÃ©cnicos': 'Aerotécnicos',
        'Ãgua Limpa Transportes': 'Água Limpa Transportes',
        'Ãngel LascurÃ¡in y Osio': 'Ángel Lascuráin y Osio',
        'Ãrzteflugambulanz': 'Ärzteflugambulanz',
        'Ãtablissements Economique du Casino': 'Établissements Economique du Casino',
        'ÃLAG': 'ÖLAG',
        'privateÂ': 'private',
        # Add more replacements as needed
    }
    df['operator'] = df['operator'].replace(replacements)
    # Strip whitespace and non-breaking spaces
    df['operator'] = df['operator'].astype(str).str.strip().str.replace('\u00A0', '', regex=False)
    # Set empty, whitespace-only, or 'unknown' (case-insensitive) to 'Unknown'
    df['operator'] = df['operator'].replace(
        to_replace=[r'^$', r'^\s*$', r'unknown', r'Unknown', None, pd.NA], 
        value='Unknown', 
        regex=True
    )
    df['operator'] = df['operator'].fillna('Unknown')
    return df

df = clean_operator(df)