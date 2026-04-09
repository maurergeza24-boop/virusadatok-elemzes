import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adat Vizualizáció", layout="wide")

st.title("Interaktív Adatvizualizáció és Trendszimuláció")

# Google Sheets elérhetőség (CSV export formátumban)
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    
    # Csak a kért oszlopok megtartása (A az idő, a többi a megadott betűk alapján)
    # L=11, O=14, S=18, T=19, U=20 indexű oszlopok (0-tól kezdve)
    selected_columns = [0, 11, 14, 18, 19, 20]
    df = df.iloc[:, selected_columns]
    
    # Oszlopok elnevezése
    df.columns = ['Dátum', 'Aktív fertőzöttek száma', 'Hatósági házi karantén', 
                  'Új gyógyultak száma', 'Kórházi ápoltak száma', 'Lélegeztetőgépen lévők száma']
    
    # Dátum konvertálása
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum'])
    
    # Numerikus értékek tisztítása és hiányzó/nulla adatok simítása interpolációval
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # A 0 értékeket NaN-ra cseréljük, hogy az interpoláció kitöltse őket
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
    return df

try:
    data = load_data()

    # Legördülő menü a szempontokhoz
    options = data.columns[1:].tolist()
    selected_col = st.selectbox("Válasszon egy szempontot a vizualizációhoz:", options)

    # Statisztikai számítások
    y_values = data[selected_col].values
    mean_val = np.mean(y_values)
    
    # Aszimmetrikus szórás számítása
    above_mean = y_values[y_values > mean_val]
    below_mean = y_values[y_values <= mean_val]
    
    std_upper = np.mean(np.abs(above_mean - mean_val)) if len(above_mean) > 0 else 0
    std_lower = np.mean(np.abs(below_mean - mean_val)) if len(below_mean) > 0 else 0

    # Szimuláció funkció
    def generate_simulation(original_data):
        # Napi változások (százalékos elmozdulás a trendhez)
        returns = np.diff(original_data) / original_data[:-1]
        returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Új sorozat generálása (Random Walk with Drift)
        simulated = [original_data[0]]
        for i in range(len(original_data)-1):
            next_val = simulated[-1] * (1 + np.random.normal(avg_return, std_return))
            simulated.append(max(0, next_val)) # Ne legyen negatív
        return np.array(simulated)

    # Új szimuláció gomb
    if 'sim_data' not in st.session_state or st.button("Új szimuláció"):
        st.session_state.sim_data = generate_simulation(y_values)

    # Grafikon készítése Plotly-val
    fig = go.Figure()

    # 1. Valós értékek
    fig.add_trace(go.Scatter(x=data['Dátum'], y=y_values, name="Valós értékek", line=dict(color='blue', width=2)))

    # 2. Átlag
    fig.add_trace(go.Scatter(x=data['Dát_um'], y=[mean_val]*len(data), name="Átlag", line=dict(color='gray', dash='dash')))

    # 3. Felső szórás
    fig.add_trace(go.Scatter(x=data['Dátum'], y=[mean_val + std_upper]*len(data), name="Felső szórás", line=dict(color='green', width=1, dash='dot')))

    # 4. Alsó szórás
    fig.add_trace(go.Scatter(x=data['Dátum'], y=[mean_val - std_lower]*len(data), name="Alsó szórás", line=dict(color='orange', width=1, dash='dot')))

    # 5. Szimulált értékek
    fig.add_trace(go.Scatter(x=data['Dátum'], y=st.session_state.sim_data, name="Szimulált értékek", line=dict(color='red', width=2)))

    fig.update_layout(
        xaxis_title="Idő",
        yaxis_title="Mennyiség",
        legend_title="Jelmagyarázat",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt az adatok feldolgozása során: {e}")
    st.info("Ellenőrizze, hogy a Google Sheet linkje publikusan elérhető-e!")
