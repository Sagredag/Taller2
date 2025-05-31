import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sala Golden", layout="wide")

# ————— 1. Carga de datos —————
st.sidebar.header("🔄 Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel/CSV", type=['xlsx','csv'])

if not uploaded_file:
    st.warning("Por favor, sube un archivo CSV desde la barra lateral para mostrar el dashboard.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ————— 2. Preprocesamiento —————
for col in ['Producción','COIN','Real','LSTM']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = 0

if 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# ————— 3. Header y métricas —————
st.title("🎰 Sala Golden")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Producción", f"S/. {df['Producción'].sum():,.0f}")
c2.metric("Producción en Dólares", f"${df['Producción'].sum() / 3.37:,.0f}")
c3.metric("COIN", f"S/. {df['COIN'].sum():,.2f}")
c4.metric("Precisión Esperada", "90%")

st.markdown("---")

# ————— 4. Filtros de fecha —————
if 'Fecha' in df.columns:
    d1, d2 = st.columns(2)
    start_date = d1.date_input("Desde", df['Fecha'].min())
    end_date   = d2.date_input("Hasta", df['Fecha'].max())
    df = df[(df['Fecha'] >= pd.to_datetime(start_date)) & (df['Fecha'] <= pd.to_datetime(end_date))]

# ————— 5. Gráfico nativo —————
st.subheader("📈 Producción Física vs Predicción (LSTM)")
chart_df = df[['Real', 'LSTM']].fillna(0)
st.line_chart(chart_df, use_container_width=True)

st.markdown("---")

# ————— 6. Tabla de detalle —————
st.subheader("📋 Datos de Máquinas — Sala Golden")

if 'Id Maquina' in df.columns:
    opciones = df['Id Maquina'].dropna().unique().astype(str).tolist()
    filtro   = st.selectbox("Filtrar por Nº Máquina", ["Todas"] + opciones)
    if filtro != "Todas":
        df = df[df['Id Maquina'].astype(str) == filtro]

cols_para_mostrar = [c for c in ['Id Maquina','Num Serie','Producción','COIN'] if c in df.columns]
st.dataframe(
    df[cols_para_mostrar]
      .sort_values(by='Producción', ascending=False)
      .reset_index(drop=True),
    use_container_width=True
)

st.markdown("---")
st.caption("Proyecto de predicción con LSTM | Taller Integrador – 2025")
