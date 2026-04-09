import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Oldal beállítása
st.set_page_config(page_title="Pandémia Adatvizualizáló", layout="wide")

st.title("Interaktív Trendvizualizáció és Szimuláció")
st.markdown("""
Ez az alkalmazás a megadott Google Sheet adatait dolgozza fel. 
A szimuláció a valós adatok mozgóátlaga és napi volatilitása alapján készül.
""")

# Adatok beöltése a Google Sheets-ből
sheet_id = "1e4VEZL1xvsALoOIq9V2SQuICeQrT5MtWfBm32ad7i8Q"
gid = "311133316"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_and_clean_data():
    # Beolvasás
    df = pd.read_csv(url)
    
    # A kért oszlopok: A(0), L(11), O(14), S(18), T(19), U(20)
    # Az oszlopok indexelése 0-tól indul
    indices = [0, 11, 14, 18, 19, 20]
    df = df.iloc[:, indices]
    
    # Oszlopok elnevezése a kérés szerint
    df.columns = [
        'Dátum', 
        'Aktív fertőzöttek száma', 
        'Hatósági házi karantén', 
        'Új gyógyultak száma', 
        'Kórházi ápoltak száma', 
        'Lélegeztetőgépen lévők száma'
    ]
    
    # Dátum formázása
    df['Dátum'] = pd.to_datetime(df['Dátum'], errors='coerce')
    df = df.dropna(subset=['Dátum'])
    df = df.sort_values('Dátum')
    
    # Adatok simítása: 0-ák és hiányzó adatok interpolálása
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # A 0-át hibás adatnak tekintjük a kérés szerint, és interpoláljuk
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        # Ha maradt NaN (pl. az elején), azt 0-ra állítjuk
        df[col] = df[col].fillna(0)
        
    return df

try:
    df = load_and_clean_data()

    # Legördülő menü az oszlopok kiválasztásához
    features = df.columns[1:].tolist()
    selected_feature = st.selectbox("Válasszon egy szempontot:", features)

    # Aktuális adatsor kinyerése
    y_real = df[selected_feature].values
    dates = df['Dátum']

    # --- Statisztikai számítások ---
    mean_val = np.mean(y_real)
    
    # Eltérés az átlagtól (Felső és Alsó szórás kérés szerint)
    diff_above = y_real[y_real > mean_val] - mean_val
    diff_below = mean_val - y_real[y_real <= mean_val]
    
    std_upper = np.mean(diff_above) if len(diff_above) > 0 else 0
    std_lower = np.mean(diff_below) if len(diff_below) > 0 else 0

    # --- Trendszimuláció Modell ---
    def run_trend_simulation(data_series):
        n = len(data_series)
        # 7 napos mozgóátlag a trend irányának meghatározásához
        rolling_trend = pd.Series(data_series).rolling(window=7, min_periods=1).mean().values
        # Napi ugrálások (volatilitás) mértéke
        daily_diffs = np.diff(data_series)
        volatility = np.std(daily_diffs) if len(daily_diffs) > 0 else 1
        
        simulated = np.zeros(n)
        simulated[0] = data_series[0]
        
        for i in range(1, n):
            # A trend faktor: merre mozdult el a mozgóátlag az előző naphoz képest
            trend_step = rolling_trend[i] - rolling_trend[i-1]
            # Véletlenszerű zaj a valós adatok ingadozása alapján
            noise = np.random.normal(0, volatility * 0.7) 
            
            # Új érték = előző szimulált érték + trend iránya + zaj
            val = simulated[i-1] + trend_step + noise
            simulated[i] = max(0, val) # Ne legyen negatív
            
        return simulated

    # Szimuláció gomb kezelése
    if 'current_sim' not in st.session_state or st.button("Új szimuláció"):
        st.session_state.current_sim = run_trend_simulation(y_real)

    # --- Vizualizáció Plotly-val ---
    fig = go.Figure()

    # 1. Valós adatok (Kék)
    fig.add_trace(go.Scatter(
        x=dates, y=y_real, name="Valós értékek",
        line=dict(color='#1f77b4', width=3)
    ))

    # 2. Átlag (Fekete szaggatott)
    fig.add_trace(go.Scatter(
        x=dates, y=[mean_val]*len(df), name="Átlag",
        line=dict(color='black', dash='dash', width=2)
    ))

    # 3. Felső szórás (Zöld pontozott)
    fig.add_trace(go.Scatter(
        x=dates, y=[mean_val + std_upper]*len(df), name="Felső szórás",
        line=dict(color='#2ca02c', dash='dot', width=1.5)
    ))

    # 4. Alsó szórás (Narancs pontozott)
    fig.add_trace(go.Scatter(
        x=dates, y=[mean_val - std_lower]*len(df), name="Alsó szórás",
        line=dict(color='#ff7f0e', dash='dot', width=1.5)
    ))

    # 5. Szimulált értékek (Piros)
    fig.add_trace(go.Scatter(
        x=dates, y=st.session_state.current_sim, name="Szimulált értékek",
        line=dict(color='#d62728', width=2)
    ))

    # Grafikon kinézetének finomhangolása
    fig.update_layout(
        height=600,
        xaxis_title="Idő (Dátum)",
        yaxis_title="Érték / Mennyiség",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )

    # Grafikon megjelenítése
    st.plotly_chart(fig, use_container_width=True)

    # Statisztikai adatok megjelenítése kártyákon (opcionális extra)
    col1, col2, col3 = st.columns(3)
    col1.metric("Átlagos érték", f"{mean_val:.2f}")
    col2.metric("Felső eltérés (átlagtól)", f"{std_upper:.2f}")
    col3.metric("Alsó eltérés (átlagtól)", f"{std_lower:.2f}")

except Exception as e:
    st.error(f"Hiba történt az adatok feldolgozása közben: {e}")
    st.info("Kérjük, ellenőrizze, hogy a Google Sheet publikusan megosztható-e a link birtokában!")
