import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal konfiguráció
st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Interaktív Trendvizualizáció és Szimuláció")
st.info("Az adatsorokat 7 napos mozgóátlaggal simítottuk a hétvégi adatközlési anomáliák (zaj) kiküszöbölése érdekében.")

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
    
    # Adattisztítás és Lokális Simítás
    for col in df.columns[1:]:
        # Karaktertisztítás (szóközök eltávolítása)
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Lépés: 0/NaN interpolálása (hiányzó adatok kitöltése)
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # 2. Lépés: Lokális simítás (7 napos ablak)
        # Ez küszöböli ki a hétvégi "zajos" kilengéseket az eredeti görbén
        df[col] = df[col].rolling(window=7, min_periods=1, center=True).mean()
    
    return df

try:
    data = load_data()
    
    # Kezelőfelület
    col_selector, col_button = st.columns([3, 1])
    
    with col_selector:
        options = data.columns[1:].tolist()
        selected_col = st.selectbox("Válasszon egy szempontot:", options)
    
    y_real = data[selected_col].astype(float).values
    dates = data['Dátum']

    # Statisztikák (A már simított napi adatok alapján)
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # Stabilizált Szimulációs Modell
    def generate_sim(original):
        n = len(original)
        # Mivel az 'original' már simított, a trend tiszta
        # A zaj mértékét a simított adatok maradék ingadozásához kötjük
        volatility = np.std(np.diff(original)) if len(original) > 1 else 1
        
        sim = np.zeros(n)
        sim[0] = original[0]
        for i in range(1, n):
            # A trendet az eredeti simított görbe napi változása adja
            trend_step = original[i] - original[i-1]
            # Kontrollált zaj a természetes hatásért
            noise = np.random.normal(0, volatility * 0.8)
            sim[i] = max(0, sim[i-1] + trend_step + noise)
        
        # A szimulációt is áteresztjük egy enyhe simítón a tiszta képért
        return pd.Series(sim).rolling(window=3, min_periods=1, center=True).mean().values

    # Állapotkezelés
    if 'sim_values' not in st.session_state or 'last_feature' not in st.session_state or st.session_state.last_feature != selected_col:
        st.session_state.sim_values = generate_sim(y_real)
        st.session_state.last_feature = selected_col

    with col_button:
        st.write("##")
        if st.button("Új szimuláció"):
            st.session_state.sim_values = generate_sim(y_real)

    # GRAFIKON
    fig = go.Figure()
    h_template = "Dátum: %{x}<br>Napi érték: %{y:,.0f}<extra></extra>"

    fig.add_trace(go.Scatter(x=dates, y=y_real, name="Valós értékek (simított)", 
                             line=dict(color='blue', width=2.5), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val]*len(data), name="Átlag", 
                             line=dict(color='black', dash='dash'), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val + std_upper]*len(data), name="Felső szórás", 
                             line=dict(color='green', dash='dot'), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=[mean_val - std_lower]*len(data), name="Alsó szórás", 
                             line=dict(color='orange', dash='dot'), hovertemplate=h_template))
    
    fig.add_trace(go.Scatter(x=dates, y=st.session_state.sim_values, name="Szimulált értékek", 
                             line=dict(color='red', width=2), hovertemplate=h_template))

    fig.update_layout(
        height=700,
        xaxis_title="Idő",
        yaxis_title="Napi mennyiség (7 napos átlag)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        yaxis=dict(tickformat=",d")
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt: {e}")
