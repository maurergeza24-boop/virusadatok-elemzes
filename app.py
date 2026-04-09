import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Intelligens Trendvizualizáció")
st.info("A rendszer automatikusan azonosítja és tompítja az adminisztratív adatközlésből adódó extrém kiugrásokat (outliers).")

# Google Sheets elérhetőség
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    indices = [0, 11, 14, 18, 19, 20]
    df = df.iloc[:, indices]
    df.columns = [
        'Dátum', 'Aktív fertőzöttek száma', 'Hatósági házi karantén', 
        'Új gyógyultak száma', 'Kórházi ápoltak száma', 'Lélegeztetőgépen lévők száma'
    ]
    
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum']).sort_values('Dátum')
    
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Alapvető kitöltés
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # 2. INTELLIGENS KIUGRÁS-SZŰRÉS (Outlier clipping)
        # Kiszámoljuk a 7 napos mozgó mediánt (ez ellenállóbb a tüskékkel szemben)
        rolling_median = df[col].rolling(window=7, center=True, min_periods=1).median()
        # Meghatározunk egy küszöböt: a medián 3-szorosa feletti értékeket tüskének tekintjük
        threshold = rolling_median * 3
        # Ahol az adat túllépi a küszöböt, ott a küszöbértékre limitáljuk
        df[col] = np.where(df[col] > threshold, threshold, df[col])
        
        # 3. Végső simítás (7 napos átlag)
        df[col] = df[col].rolling(window=7, min_periods=1, center=True).mean()
    
    return df

try:
    data = load_data()
    
    col_selector, col_button = st.columns([3, 1])
    with col_selector:
        selected_col = st.selectbox("Válasszon egy szempontot:", data.columns[1:].tolist())
    
    y_real = data[selected_col].astype(float).values
    dates = data['Dátum']

    # Statisztikák
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # Stabil szimuláció (zajcsökkentett)
    def generate_sim(original):
        n = len(original)
        # Csökkentett volatilitás a simább szimulációért
        volatility = np.std(np.diff(original)) * 0.4 if len(original) > 1 else 1
        sim = np.zeros(n)
        sim[0] = original[0]
        for i in range(1, n):
            trend_step = original[i] - original[i-1]
            noise = np.random.normal(0, volatility)
            sim[i] = max(0, sim[i-1] + trend_step + noise)
        return pd.Series(sim).rolling(window=3, min_periods=1, center=True).mean().values

    if 'sim_values' not in st.session_state or 'last_feature' not in st.session_state or st.session_state.last_feature != selected_col:
        st.session_state.sim_values = generate_sim(y_real)
        st.session_state.last_feature = selected_col

    with col_button:
        st.write("##")
        if st.button("Új szimuláció"):
            st.session_state.sim_values = generate_sim(y_real)

    # Grafikon
    fig = go.Figure()
    h_template = "Dátum: %{x}<br>Érték: %{y:,.0f}<extra></extra>"

    fig.add_trace(go.Scatter(x=dates, y=y_real, name="Valós értékek (szűrt)", line=dict(color='blue', width=2.5), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val]*len(data), name="Átlag", line=dict(color='black', dash='dash'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val + std_upper]*len(data), name="Felső szórás", line=dict(color='green', dash='dot'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val - std_lower]*len(data), name="Alsó szórás", line=dict(color='orange', dash='dot'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=st.session_state.sim_values, name="Szimulált értékek", line=dict(color='red', width=2), hovertemplate=h_template))

    fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template="plotly_white", yaxis=dict(tickformat=",d"))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt: {e}")
