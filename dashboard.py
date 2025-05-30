import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sala Golden", layout="wide")

# ————— 1. Carga y limpieza de datos —————
@st.cache_data
def load_and_clean(file):
    # Soportar CSV y Excel
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine='openpyxl')
    # Igualar formato Colab
    df = df.iloc[1:].copy()
    df.columns = [
        "Nº", "Tipo", "Fecha", "Desconocido1",
        "Entrada_Electr", "Salida_Electr", "P_Manual", "CC_Manual", "Billetero",
        "Entrada_TITO", "Salida_TITO"
    ]
    # Coerción de tipos
    for c in ["Entrada_Electr","Salida_Electr","Billetero"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
    df = df.dropna(subset=["Entrada_Electr","Salida_Electr","Billetero"]).reset_index(drop=True)
    # Corrección de reinicios
    def corregir_reset(s, cap=99999999):
        s = s.astype(float).reset_index(drop=True)
        difs = [0]
        for i in range(1, len(s)):
            if s[i] >= s[i-1]:
                difs.append(s[i] - s[i-1])
            else:
                difs.append((cap - s[i-1]) + s[i] + 1)
        return pd.Series(difs)
    df['Prod_Electr'] = corregir_reset(df['Entrada_Electr']) - corregir_reset(df['Salida_Electr'])
    df['Prod_Billetero'] = corregir_reset(df['Billetero'])
    return df

# ————— 2. Secuencias —————
@st.cache_data
def create_sequences(signal, n_steps=10):
    X, y = [], []
    for i in range(len(signal) - n_steps):
        X.append(signal[i:i+n_steps])
        y.append(signal[i+n_steps])
    return np.array(X), np.array(y)

# ————— 3. Entrenamiento modelos —————
@st.cache_resource
def train_models(df, n_steps=10):
    # Normalización
    scaler = MinMaxScaler()
    signal = df[['Prod_Billetero']].values
    signal_scaled = scaler.fit_transform(signal)
    # Crear secuencias
    X, y = create_sequences(signal_scaled, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Para MLP
    X_mlp = X.reshape(X.shape[0], -1)
    X_tr_mlp, X_te_mlp, y_tr_mlp, y_te_mlp = train_test_split(X_mlp, y, test_size=0.2, shuffle=False)
    # LSTM
    model_lstm = Sequential([LSTM(64, input_shape=(n_steps, 1)), Dense(1)])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    # GRU
    model_gru = Sequential([GRU(64, input_shape=(n_steps, 1)), Dense(1)])
    model_gru.compile(optimizer='adam', loss='mse')
    model_gru.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    # MLP
    model_mlp = Sequential([Dense(64, activation='relu', input_shape=(X_tr_mlp.shape[1],)), Dense(1)])
    model_mlp.compile(optimizer='adam', loss='mse')
    model_mlp.fit(X_tr_mlp, y_tr_mlp, epochs=20, batch_size=32, verbose=0)
    # ARIMA
    train_series = signal_scaled[:len(X_train)+n_steps].flatten()
    arima_model = ARIMA(train_series, order=(1,0,1)).fit()
    arima_pred_scaled = arima_model.forecast(steps=len(X_test))
    arima_pred = scaler.inverse_transform(arima_pred_scaled.reshape(-1,1)).flatten()
    return {
        'scaler': scaler,
        'n_steps': n_steps,
        'X_test': X_test,
        'y_test': y_test,
        'models': {'LSTM': model_lstm, 'GRU': model_gru, 'MLP': model_mlp},
        'arima_pred': arima_pred
    }

# ————— 4. Métricas —————
@st.cache_data
def compute_metrics(tr):
    scaler = tr['scaler']
    X_test = tr['X_test']
    y_test = tr['y_test']
    def desnorm(y_s): return scaler.inverse_transform(y_s.reshape(-1,1)).flatten()
    results = []
    for name, model in tr['models'].items():
        if name == 'MLP':
            X_in = X_test.reshape(X_test.shape[0], -1)
        else:
            X_in = X_test
        y_pred_s = model.predict(X_in).flatten()
        y_true = desnorm(y_test)
        y_hat = desnorm(y_pred_s)
        results.append([
            name,
            mean_absolute_error(y_true, y_hat),
            np.sqrt(mean_squared_error(y_true, y_hat)),
            (1 - mean_absolute_percentage_error(y_true[y_true!=0], y_hat[y_true!=0])) * 100,
            r2_score(y_true, y_hat) * 100
        ])
    # ARIMA
    y_true = desnorm(y_test)
    arima_pred = tr['arima_pred']
    results.append([
        'ARIMA',
        mean_absolute_error(y_true, arima_pred),
        np.sqrt(mean_squared_error(y_true, arima_pred)),
        (1 - mean_absolute_percentage_error(y_true[y_true!=0], arima_pred[y_true!=0])) * 100,
        r2_score(y_true, arima_pred) * 100
    ])
    return pd.DataFrame(results, columns=['Modelo','MAE','RMSE','Accuracy_%','R2_%'])

# ————— 5. Interfaz —————
st.sidebar.header("🔄 Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel/CSV", type=['xlsx','csv'])
if not uploaded_file:
    st.warning("Por favor, sube tu archivo para continuar.")
    st.stop()

df = load_and_clean(uploaded_file)

# Métricas globales
st.title("🎰 Sala Golden")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Billetes", f"{df['Prod_Billetero'].sum():,.0f}")
c2.metric("Total Producción Eléctrica", f"{df['Prod_Electr'].sum():,.0f}")
c3.metric("Período", f"{df['Fecha'].min().date()} a {df['Fecha'].max().date()}")
c4.metric("Modelos", "LSTM, GRU, MLP, ARIMA")

st.markdown("---")
with st.spinner("Entrenando modelos..."):
    tr = train_models(df)

# Mostrar métricas de evaluación
metrics_df = compute_metrics(tr)
st.subheader("📊 Métricas de evaluación")
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")
# Comparativa Real vs Predicción
st.subheader("📈 Real vs Predicciones en Test Set")
scaler = tr['scaler']
y_true = scaler.inverse_transform(tr['y_test'].reshape(-1,1)).flatten()
preds = {}
for m in ['LSTM','GRU','MLP']:
    preds[m] = scaler.inverse_transform(
        tr['models'][m].predict(
            tr['X_test'] if m!='MLP' else tr['X_test'].reshape(tr['X_test'].shape[0],-1)
        )
        .flatten().reshape(-1,1)
    ).flatten()
preds['ARIMA'] = tr['arima_pred']
plot_df = pd.DataFrame({'Real': y_true, **preds})
st.line_chart(plot_df)

st.markdown("---")
# Agregación diaria
test_start = len(df) - len(tr['X_test']) - tr['n_steps']
test_dates = df['Fecha'].iloc[test_start + tr['n_steps']:test_start + tr['n_steps'] + len(tr['X_test'])]
test_dates = test_dates.dt.date.reset_index(drop=True)
daily = pd.DataFrame({'Fecha': test_dates, 'Real': y_true, **preds})
daily = daily.groupby('Fecha').sum()
fig, ax = plt.subplots()
for col in daily.columns:
    ax.plot(daily.index, daily[col], label=col)
ax.set_title('Producción neta diaria vs predicha')
ax.set_xlabel('Fecha')
ax.set_ylabel('Billetes / día')
ax.legend()
st.pyplot(fig)

st.markdown("---")
# Matriz de confusión LSTM
st.subheader("📊 Matriz de Confusión: Real vs LSTM")
# Definir bins por cuantiles
q1 = daily['Real'].quantile(1/3)
q2 = daily['Real'].quantile(2/3)
if q1 == q2:
    bins = [-np.inf, q1, np.inf]; labels = ['Bajo','Alto']
else:
    bins = [-np.inf, q1, q2, np.inf]; labels = ['Bajo','Medio','Alto']
real_cat = pd.cut(daily['Real'], bins=bins, labels=labels, duplicates='drop')
pred_cat = pd.cut(daily['LSTM'], bins=bins, labels=labels, duplicates='drop')
conf_mat = pd.crosstab(real_cat, pred_cat, rownames=['Real'], colnames=['Predicción LSTM'], margins=True, margins_name='Total')
st.write(conf_mat)
fig2, ax2 = plt.subplots()
sns.heatmap(conf_mat.iloc[:-1,:-1], annot=True, fmt='d', ax=ax2)
ax2.set_title('Confusión LSTM')
st.pyplot(fig2)

st.caption("Proyecto de predicción con LSTM, GRU, MLP y ARIMA | Taller Integrador – 2025")
