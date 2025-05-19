import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# -----------------------------------------------------------------------------
# Declare some useful functions.
df = pd.read_csv('data/data_inflasi.csv')
df['Year'] = pd.to_datetime(df['Year'], format='%b %Y')

def create_sequences(data, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

def arima_lstm_pred(data):
    prg = st.progress(0, text="Operation in progress. Please wait.")
    data = data['Inflasi'].values
    prg.progress(10, text="Memproses data...")

    # Step 1: Fit ARIMA model
    arima_order = (3, 1, 4)
    arima_model = ARIMA(data, order=arima_order)
    arima_fit = arima_model.fit()
    prg.progress(30, text="Model ARIMA dilatih...")

    # Tampilkan summary ARIMA
    st.subheader("Ringkasan Model ARIMA")
    st.text(arima_fit.summary().as_text())

    # Prediksi ARIMA
    arima_pred = arima_fit.predict(start=0, end=len(data)-1, typ='levels')
    residuals = data - arima_pred

    # Split data
    train_size = int(len(data) * 0.9)
    data_training = data[:train_size]
    data_testing = data[train_size:]

    # Normalisasi
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled_training = scaler.fit_transform(data_training.reshape(-1, 1))
    data_scaled_testing = scaler.transform(data_testing.reshape(-1, 1))
    prg.progress(50, text="Data dinormalisasi...")

    # Sequence
    timesteps = 10
    X_train, y_train = create_sequences(data_scaled_training, timesteps)
    X_test, y_test = create_sequences(data_scaled_testing, timesteps)

    # Build LSTM
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    prg.progress(70, text="Model LSTM dibangun...")

    # Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train LSTM
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    prg.progress(85, text="Model LSTM dilatih...")

    # Tampilkan summary LSTM
    st.subheader("Ringkasan Arsitektur Model LSTM")
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    st.text('\n'.join(summary_list))

    # Tampilkan grafik loss history
    st.subheader("Grafik Training vs Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    st.pyplot(fig)

    # Forecast bulan depan
    future_steps = 1
    future_inputs = data_scaled_testing[-timesteps:]
    future_predictions_lstm = []

    for _ in range(future_steps):
        input_seq = future_inputs.reshape(1, timesteps, 1)
        lstm_pred_scaled = model.predict(input_seq, verbose=0)
        future_predictions_lstm.append(lstm_pred_scaled[0, 0])
        future_inputs = np.append(future_inputs[1:], lstm_pred_scaled, axis=0)

    future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1)).flatten()
    future_predictions_arima = arima_fit.forecast(steps=future_steps)

    final_forecast = future_predictions_arima + future_predictions_lstm
    prg.progress(100, text="Prediksi selesai.")

    st.subheader("Hasil Prediksi Inflasi Bulan Berikutnya")
    st.markdown(f"<h1 style='text-align: center;'>{round(final_forecast[0], 2)}%</h1>", unsafe_allow_html=True)

    return round(final_forecast[0], 2)

# -----------------------------------------------------------------------------
# Draw the actual page
'''
# Prediksi Inflasi Bulan Selanjutnya
'''
uploaded_file = st.file_uploader("Masukkan data inflasi", type="csv")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    result = arima_lstm_pred(dataframe)
    st.markdown(f"<h1 style='text-align: center;'>{result}%</h1>", unsafe_allow_html=True)
