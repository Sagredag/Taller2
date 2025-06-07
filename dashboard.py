import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Función para corregir reinicios de contador
def corregir_diferencias_con_reset(serie, cap_valor=99999999):
    s = serie.astype(float).reset_index(drop=True)
    difs = [0.0]
    for i in range(1, len(s)):
        if s[i] >= s[i-1]:
            difs.append(s[i] - s[i-1])
        else:
            difs.append((cap_valor - s[i-1]) + s[i] + 1)
    return np.array(difs)

st.title("Producción vs Predicción con LSTM")

# Subida de archivo
datos = st.file_uploader("Sube tu archivo Excel", type=["xls","xlsx"])
if datos:
    # 1) Lectura y selección de columnas relevantes
    df = pd.read_excel(datos, engine='openpyxl')
    df = df.iloc[1:].copy()
    df.columns = ["N°","Tipo","Fecha","Desconocido","Entrada","Salida","P_Manual","CC_Manual","Billetero","Entrada_TITO","Salida_TITO"]
    df = df[["Fecha","Entrada","Salida"]]

    # 2) Conversión de tipos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
    # Opción de unidades: centavos o soles
    to_soles = st.checkbox("Convertir a soles (x0.01)", value=True)
    factor = 0.01 if to_soles else 1.0
    df["Entrada"] = pd.to_numeric(df["Entrada"], errors='coerce') * factor
    df["Salida"]  = pd.to_numeric(df["Salida"],  errors='coerce') * factor
    df = df.dropna(subset=["Fecha","Entrada","Salida"]).reset_index(drop=True)

    # 3) Cálculo de producción (Entrada - Salida) con corrección de resets
    diff_ent = corregir_diferencias_con_reset(df["Entrada"])
    diff_sal = corregir_diferencias_con_reset(df["Salida"])
    df["Produccion"] = diff_ent - diff_sal

    # Mostrar tabla completa de producción por fecha
    st.subheader("Producción por bloque de tiempo")
    st.dataframe(df[["Fecha","Produccion"]])

    # 4) Normalización de la señal
    signal = df[["Produccion"]].values
    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal)

    # 5) Creación de secuencias
    def create_sequences(data, n_steps=10):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = st.slider("Pasos para LSTM", min_value=5, max_value=50, value=10)
    X, y = create_sequences(signal_scaled, n_steps)

    # 6) División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 7) Definición y entrenamiento del modelo LSTM
    epochs = st.number_input("Épocas", min_value=1, max_value=100, value=20)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32)
    model = Sequential([LSTM(50, input_shape=(n_steps, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    with st.spinner('Entrenando LSTM...'):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    st.success('Entrenamiento completado')

    # 8) Predicciones para todo el conjunto de datos
    y_full_pred = model.predict(np.vstack([X_train, X_test])).flatten()
    # Desnormalización
    delta = scaler.data_max_[0] - scaler.data_min_[0]
    y_true_full = signal[n_steps:].flatten()
    y_pred_full = y_full_pred * delta + scaler.data_min_[0]

    # 9) Preparar DataFrame de predicciones
    df_pred = pd.DataFrame({
        'Fecha': df['Fecha'].iloc[n_steps:].reset_index(drop=True),
        'Real': y_true_full,
        'Predicho': y_pred_full
    })

    # Agregar diaria (resample)
    df_daily = df_pred.set_index('Fecha').resample('D').sum()
    # Asegurar todas las fechas
    all_dates = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq='D')
    df_daily = df_daily.reindex(all_dates, fill_value=0)

    # 10) Gráfica diaria (más fácil de leer)
    st.subheader("Producción neta diaria vs predicha")
    st.line_chart(df_daily)

    # 11) Gráfica por bloque (opcional)
    if st.checkbox('Mostrar producción por bloque'):
        st.subheader('Producción Real vs Predicha por bloque')
        st.line_chart(df_pred.set_index('Fecha')[['Real','Predicho']])
