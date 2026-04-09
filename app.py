import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Interaktív Trendvizualizáció és Szimuláció")
st.info("A szimulált görbe a valós adatok 7 napos mozgóátlaga és csökkentett zajfaktor alapján készül a stabilabb megjelenítés érdekében.")

# Google Sheets elérhetőség
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    # Oszlopok kiválasztása: A(0), L(11), O(14), S(18), T(19), U(20)
    indices = [0, 11, 14, 18, 19, 20]
    df = df.iloc[:, indices]
    df.columns = [
        'Dátum', 'Aktív fertőzöttek száma', 'Hatósági házi karantén', 
        'Új gyógyultak száma', 'Kórházi ápoltak száma', 'Lélegeztetőgépen lévők száma'
    ]
    
    # Dátum tisztítása
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum']).sort_values('Dátum')
    
    # Adattisztítás
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Interpoláció a 0/NaN értékekre a görbe simítása érdekében
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
    
    return df

try:
    data = load_data()
    
    # Kezelőfelület
    col_selector, col_button = st.columns([3, 1])
    
    with col_selector:
        options = data.columns[1:].tolist()
        selected_col = st.selectbox("Válasszon egy szempontot:", options)
    
    # Adatok előkészítése
    y_real = data[selected_col].astype(float).values
    dates = data['Dátum']

    # Statisztikák (Valós napi adatok alapján)
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # SIMÍTOTT TRENDSZIMULÁCIÓ
    def generate_sim(original):
        n = len(original)
        # 7 napos mozgóátlag a stabil trendért
        rolling_trend = pd.Series(original).rolling(window=7, min_periods=1, center=True).mean().values
        
        # Volatilitás (napi zaj) mérése és jelentős csökkentése (30% az eredetihez képest)
        diffs = np.diff(original)
        volatility = np.std(diffs) * 0.3 if len(diffs) > 1 else 1
        
        sim = np.zeros(n)
        sim[0] = original[0]
        
        for i in range(1, n):
            # A trendváltozást a mozgóátlag különbsége adja (nem a nyers napi adat)
            trend_step = rolling_trend[i] - rolling_trend[i-1]
            # Csökkentett zaj hozzáadása
            noise = np.random.normal(0, volatility)
            sim[i] = max(0, sim[i-1] + trend_step + noise)
        
        # Utólagos 3 napos simítás a szimulált görbén a vizuális zajmentességért
        return pd.Series(sim).rolling(window=3, min_periods=1, center=True).
