import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import io

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'value' not in df.columns:
        raise ValueError("CSV must contain a 'value' column.")
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(inplace=True)
    return df

def generate_dummy_data(rows=100):
    np.random.seed(42)
    return pd.DataFrame({'value': np.random.uniform(1.01, 5.00, size=rows)})

def plot_data(df):
    st.line_chart(df['value'], use_container_width=True)

def train_arima_model(df, order=(5, 1, 0)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit):
    forecast = model_fit.forecast(steps=1)
    return forecast[0]

def prepare_lstm_data(series, n_steps=10):
    X, y = [], []
    for i in range(n_steps, len(series)):
        X.append(series[i - n_steps:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm_model(data, n_steps=10, epochs=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = prepare_lstm_data(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    last_sequence = scaled_data[-n_steps:].reshape((1, n_steps, 1))
    prediction = model.predict(last_sequence, verbose=0)
    return scaler.inverse_transform(prediction)[0][0]

def export_predictions(arima_pred, lstm_pred):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"[{now}] ARIMA: {arima_pred:.2f}x | LSTM: {lstm_pred:.2f}x\n"
    return io.StringIO(content)

st.set_page_config(page_title="Aviator Predictor", layout="centered")
st.title("🎰 Aviator Analyzer - Kali GPT")
st.caption("⛔ لأغراض تعليمية وبحثية فقط - لا تستخدمه للمراهنة")

option = st.radio("اختر مصدر البيانات:", ["📂 تحميل ملف CSV", "🧪 استخدام بيانات وهمية"])

if option == "📂 تحميل ملف CSV":
    uploaded_file = st.file_uploader("ارفع ملف يحتوي على عمود 'value'", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("✔️ تم تحميل البيانات بنجاح")
elif option == "🧪 استخدام بيانات وهمية":
    df = generate_dummy_data()
    st.success("✔️ تم توليد بيانات وهمية")

if 'df' in locals():
    with st.expander("📊 عرض البيانات"):
        st.dataframe(df.tail(20))
    plot_data(df)

    with st.spinner("🔄 جاري حساب التوقعات..."):
        arima_model = train_arima_model(df)
        arima_pred = forecast_arima(arima_model)
        lstm_pred = train_lstm_model(df['value'].values)

    st.subheader("🔮 التوقعات:")
    st.metric("ARIMA", f"{arima_pred:.2f}x")
    st.metric("LSTM", f"{lstm_pred:.2f}x")

    output = export_predictions(arima_pred, lstm_pred)
    st.download_button("💾 تحميل التوقعات كملف نصي", output, file_name="aviator_predictions.txt")
