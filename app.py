import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Interaktív Trendvizualizáció és Szimuláció")

# Google Sheets elérhetőség
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    # Oszlopok: A(0), L(11), O(14), S(18), T(19), U(20)
    indices = [0, 11, 14, 18, 19, 20]
    df = df.iloc[:, indices]
    df.columns = [
        'Dátum', 'Aktív fertőzöttek száma', 'Hatósági házi karantén', 
        'Új gyógyultak száma', 'Kórházi ápoltak száma', 'Lélegeztetőgépen lévők száma'
    ]
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum']).sort_values('Dátum')
    
    # Adatsimítás (0 és hiányzó adatok interpolálása)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
    return df

try:
    data = load_data()
    
    # Interaktív elemek: Kiválasztás és Gomb egy sorban
    col_selector, col_button = st.columns([3, 1])
    
    with col_selector:
        options = data.columns[1:].tolist()
        selected_col = st.selectbox("Válasszon egy szempontot:", options)
    
    y_real = data[selected_col].values
    dates = data['Dátum']

    # Statisztikák
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # Trendszimulációs algoritmus
    def generate_sim(original):
        n = len(original)
        rolling = pd.Series(original).rolling(window=7, min_periods=1).mean().values
        volatility = np.std(np.diff(original)) if len(original) > 1 else 1
        sim = np.zeros(n)
        sim[0] = original[0]
        for i in range(1, n):
            trend = rolling[i] - rolling[i-1]
            noise = np.random.normal(0, volatility * 0.6)
            sim[i] = max(0, sim[i-1] + trend + noise)
        return sim

    # GOMB kezelése: Itt kényszerítjük az újraszámolást
    if 'sim_values' not in st.session_state:
        st.session_state.sim_values = generate_sim(y_real)

    with col_button:
        st.write("##") # Igazítás a selectboxhoz
        if st.button("Új szimuláció"):
            st.session_state.sim_values = generate_sim(y_real)

    # GRAFIKON (Plotly)
    fig = go.Figure()

    # Adatsorok hozzáadása
    # A hovertemplate beállítása biztosítja, hogy a napi pontos értéket lássuk
    common_hover = "Dátum: %{x}<br>Érték: %{y:.0f}<extra></extra>"

    fig.add_trace(go.Scatter(x=dates, y=y_real, name="Valós értékek", 
                             line=dict(color='blue', width=2), hovertemplate=common_hover))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val]*len(data), name="Átlag", 
                             line=dict(color='black', dash='dash'), hovertemplate=common_hover))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val + std_upper]*len(data), name="Felső szórás", 
                             line=dict(color='green', dash='dot'), hovertemplate=common_hover))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val - std_lower]*len(data), name="Alsó szórás", 
                             line=dict(color='orange', dash='dot'), hovertemplate=common_hover))
    
    fig.add_trace(go.Scatter(x=dates, y=st.session_state.sim_values, name="Szimulált értékek", 
                             line=dict(color='red', width=2), hovertemplate=common_hover))

    fig.update_layout(
        xaxis_title="Idő",
        yaxis_title="Mennyiség",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt: {e}")
