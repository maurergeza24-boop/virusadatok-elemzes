import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Interaktív Trendvizualizáció")
st.info("A szkript a megadott négy kulcsfontosságú mutatót elemzi, automatikus zajszűréssel és trendszimulációval.")

# Google Sheets elérhetőség
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    # A kért oszlopok indexei: A(0), L(11), O(14), T(19), U(20)
    # S(18) - Új gyógyultak száma kihagyva
    indices = [0, 11, 14, 19, 20]
    df = df.iloc[:, indices]
    df.columns = [
        'Dátum', 
        'Aktív fertőzöttek száma', 
        'Hatósági házi karantén', 
        'Kórházi ápoltak száma', 
        'Lélegeztetőgépen lévők száma'
    ]
    
    # Dátum tisztítása
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum']).sort_values('Dátum')
    
    # Adattisztítás és zajmentesítés
    for col in df.columns[1:]:
        # Karaktertisztítás
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Alapvető interpoláció (0/NaN kezelés)
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # 2. Intelligens kiugrás-szűrés (Outlier clipping)
        # 7 napos mozgó medián alapú limitálás a technikai tüskék ellen
        rolling_median = df[col].rolling(window=7, center=True, min_periods=1).median()
        threshold = rolling_median * 3
        df[col] = np.where(df[col] > threshold, threshold, df[col])
        
        # 3. 7 napos mozgóátlagos simítás
        df[col] = df[col].rolling(window=7, min_periods=1, center=True).mean()
    
    return df

try:
    data = load_data()
    
    # Kezelőfelület
    col_selector, col_button = st.columns([3, 1])
    
    with col_selector:
        # Csak a tisztított 4 oszlop jelenik meg a választóban
        options = data.columns[1:].tolist()
        selected_col = st.selectbox("Válasszon egy szempontot:", options)
    
    y_real = data[selected_col].astype(float).values
    dates = data['Dátum']

    # Statisztikák
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # Trendszimuláció
    def generate_sim(original):
        n = len(original)
        # Volatilitás számítása a simított adatokból
        volatility = np.std(np.diff(original)) * 0.4 if len(original) > 1 else 1
        
        sim = np.zeros(n)
        sim[0] = original[0]
        for i in range(1, n):
            trend_step = original[i] - original[i-1]
            noise = np.random.normal(0, volatility)
            sim[i] = max(0, sim[i-1] + trend_step + noise)
            
        return pd.Series(sim).rolling(window=3, min_periods=1, center=True).mean().values

    # Állapotkezelés a szimulációhoz
    if 'sim_values' not in st.session_state or 'last_feature' not in st.session_state or st.session_state.last_feature != selected_col:
        st.session_state.sim_values = generate_sim(y_real)
        st.session_state.last_feature = selected_col

    with col_button:
        st.write("##")
        if st.button("Új szimuláció"):
            st.session_state.sim_values = generate_sim(y_real)

    # GRAFIKON (Plotly)
    fig = go.Figure()
    h_template = "Dátum: %{x}<br>Érték: %{y:,.0f}<extra></extra>"

    # Görbék
    fig.add_trace(go.Scatter(x=dates, y=y_real, name="Valós értékek (szűrt)", 
                             line=dict(color='#1f77b4', width=2.5), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val]*len(data), name="Átlag", 
                             line=dict(color='black', dash='dash', width=1.5), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val + std_upper]*len(data), name="Felső eltérés", 
                             line=dict(color='#2ca02c', dash='dot', width=1), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val - std_lower]*len(data), name="Alsó eltérés", 
                             line=dict(color='#ff7f0e', dash='dot', width=1), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=st.session_state.sim_values, name="Szimulált értékek", 
                             line=dict(color='#d62728', width=2), hovertemplate=h_template))

    fig.update_layout(
        height=700,
        xaxis_title="Idő",
        yaxis_title="Napi mennyiség",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        yaxis=dict(tickformat=",d")
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt az adatok betöltésekor: {e}")
