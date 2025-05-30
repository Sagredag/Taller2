import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Cargar los datos
df = pd.read_csv("datos_maquinas.csv")

# Preprocesamiento simple
df['Producción'] = pd.to_numeric(df['Producción'], errors='coerce')
df['COIN'] = pd.to_numeric(df['COIN'], errors='coerce')
df['Real'] = pd.to_numeric(df.get('Real', pd.Series([0]*len(df))), errors='coerce')
df['LSTM'] = pd.to_numeric(df.get('LSTM', pd.Series([0]*len(df))), errors='coerce')

# Título principal
st.title("🎰 Sala Golden")

# Barra superior de métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Producción", f"S/. {df['Producción'].sum():,.0f}")
col2.metric("Producción en Dólares", f"${df['Producción'].sum() / 3.37:,.0f}")
col3.metric("COIN", f"S/. {df['COIN'].sum():,.2f}")
col4.metric("Precisión Esperada", "90%")

st.markdown("---")

# Fecha (placeholder visual, no interactivo aún)
col_fecha1, col_fecha2 = st.columns(2)
col_fecha1.date_input("Desde", pd.to_datetime("2024-09-05"))
col_fecha2.date_input("Hasta", pd.to_datetime("2025-09-05"))

# Gráfico de producción real vs predicho
st.subheader("📈 Producción Física - LSTM")
fig = go.Figure()
fig.add_trace(go.Scatter(y=df['Real'], mode='lines', name='Real'))
fig.add_trace(go.Scatter(y=df['LSTM'], mode='lines', name='Predicho (LSTM)'))
fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

# Tabla de datos de máquinas
st.subheader("📋 Datos de Máquinas - Sala Golden")

# Selector de número de máquina
if 'Id Maquina' in df.columns:
    opciones = df['Id Maquina'].dropna().unique()
    seleccion = st.selectbox("Filtrar por Nº Máquina", options=["Todas"] + list(opciones.astype(str)))
    if seleccion != "Todas":
        df = df[df['Id Maquina'].astype(str) == seleccion]

# Tabla de datos
st.dataframe(df[['Id Maquina', 'Num Serie', 'Producción', 'COIN']].sort_values(by='Producción', ascending=False),
             use_container_width=True)

# Pie de página
st.markdown("---")
st.caption("Proyecto de predicción con LSTM | Taller Integrador - 2025")
