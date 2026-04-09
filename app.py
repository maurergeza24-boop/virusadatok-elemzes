import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Pandémia Adatvizualizáció", layout="wide")

st.title("Stabil Trendvizualizáció és Szimuláció")

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
        df[col] = df[col].replace(0, np.nan)
        # Lineáris interpoláció a lyukak kitöltésére
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
    return df

try:
    data = load_data()
    col_selector, col_button = st.columns([3, 1])
    
    with col_selector:
        options = data.columns[1:].tolist()
        selected_col = st.selectbox("Válasszon egy szempontot:", options)
    
    y_real = data[selected_col].astype(float).values
    dates = data['Dátum']

    # --- STATISZTIKÁK ---
    mean_val = np.mean(y_real)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # --- ÚJ, STABIL SZIMULÁCIÓS MODELL ---
    def generate_stable_sim(original):
        # 1. Kiszámoljuk a 14 napos mozgóátlagot (ez a "tiszta" trend)
        trend_line = pd.Series(original).rolling(window=14, min_periods=1, center=True).mean().values
        
        # 2. Megmérjük az eredeti adat átlagos eltérését a trendtől (ez a zaj szintje)
        noise_level = np.std(original - trend_line)
        
        # 3. A szimuláció: a tiszta trendvonal + egy kontrollált véletlen zaj
        # Nem egymásra építjük az értékeket, így nincs halmozódó hiba!
        simulated = trend_line + np.random.normal(0, noise_level * 0.4, size=len(original))
        
        # 4. Egy utolsó simítás a szimulált adatsoron a "fűrészfog" ellen
        return pd.Series(simulated).rolling(window=5, min_periods=1, center=True).mean().values

    if 'sim_values' not in st.session_state or 'last_feature' not in st.session_state or st.session_state.last_feature != selected_col:
        st.session_state.sim_values = generate_stable_sim(y_real)
        st.session_state.last_feature = selected_col

    with col_button:
        st.write("##")
        if st.button("Új szimuláció"):
            st.session_state.sim_values = generate_stable_sim(y_real)

    # --- GRAFIKON ---
    fig = go.Figure()
    h_template = "Dátum: %{x}<br>Érték: %{y:,.0f}<extra></extra>"

    fig.add_trace(go.Scatter(x=dates, y=y_real, name="Valós értékek", line=dict(color='#1f77b4', width=2), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val]*len(data), name="Átlag", line=dict(color='black', dash='dash'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val + std_upper]*len(data), name="Felső szórás", line=dict(color='#2ca02c', dash='dot'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=[mean_val - std_lower]*len(data), name="Alsó szórás", line=dict(color='#ff7f0e', dash='dot'), hovertemplate=h_template))
    fig.add_trace(go.Scatter(x=dates, y=st.session_state.sim_values, name="Szimulált értékek", line=dict(color='#d62728', width=2), hovertemplate=h_template))

    fig.update_layout(
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        yaxis=dict(tickformat=",d")
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt: {e}")
