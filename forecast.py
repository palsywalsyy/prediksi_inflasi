import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -----------------------------------------------------------------------------
# Declare some useful functions.
def create_sequences(data, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

def arima_lstm_pred(data):
    prg = st.progress(0, text="Operation in progress. Please wait.")
    
    # Add display of input data
    st.subheader("Data Inflasi yang Diinput")
    st.dataframe(data)
    
    # Create a plot of input data
    st.subheader("Grafik Data Inflasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Year' in data.columns:
        ax.plot(data['Year'], data['Inflasi'])
        ax.set_xlabel('Tahun')
    else:
        ax.plot(data['Inflasi'])
        ax.set_xlabel('Periode')
    ax.set_ylabel('Inflasi (%)')
    ax.set_title('Data Inflasi')
    ax.grid(True)
    st.pyplot(fig)
    
    data_inflasi = data['Inflasi'].values
    prg.progress(10, text="Memproses data...")

    # Step 1: Fit ARIMA model
    arima_order = (3, 1, 4)
    arima_model = ARIMA(data_inflasi, order=arima_order)
    arima_fit = arima_model.fit()
    prg.progress(30, text="Model ARIMA dilatih...")

    # Tampilkan summary ARIMA
    st.subheader("Ringkasan Model ARIMA (3,1,4)")
    st.text(arima_fit.summary().as_text())

    # Prediksi ARIMA
    arima_pred = arima_fit.predict(start=0, end=len(data_inflasi)-1, typ='levels')
    residuals = data_inflasi - arima_pred

    # Split data
    train_size = int(len(data_inflasi) * 0.9)
    data_training = data_inflasi[:train_size]
    data_testing = data_inflasi[train_size:]

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
    
    # Display LSTM architecture
    st.subheader("Arsitektur Model LSTM")
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    st.text('\n'.join(summary_list))

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
    future_inputs = data_scaled_testing[-timesteps:].reshape(-1, 1)
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

    # Display the prediction components
    st.subheader("Komponen Prediksi")
    component_data = {
        "Model": ["ARIMA", "LSTM", "Hybrid (ARIMA+LSTM)"],
        "Prediksi (%)": [
            round(future_predictions_arima[0], 2),
            round(future_predictions_lstm[0], 2),
            round(final_forecast[0], 2)
        ]
    }
    st.table(pd.DataFrame(component_data))

    return round(final_forecast[0], 2)

# -----------------------------------------------------------------------------
# Draw the actual page
st.title("Prediksi Inflasi Bulan Selanjutnya")
st.markdown("""
    Aplikasi ini menggunakan model hybrid ARIMA dan LSTM untuk memprediksi inflasi
    bulan selanjutnya berdasarkan data historis. Upload file CSV yang berisi data inflasi.
""")

st.sidebar.header("Informasi")
st.sidebar.info("""
    Model yang digunakan:
    - ARIMA (3,1,4)
    - LSTM dengan 2 layer, masing-masing 64 neuron
""")

uploaded_file = st.file_uploader("Masukkan data inflasi", type="csv")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    try:
        dataframe = pd.read_csv(uploaded_file)
        if 'Inflasi' not in dataframe.columns:
            st.error("File CSV harus memiliki kolom 'Inflasi'.")
        else:
            # Convert Year column to datetime if exists
            if 'Year' in dataframe.columns:
                dataframe['Year'] = pd.to_datetime(dataframe['Year'], format='%b %Y', errors='coerce')
                if dataframe['Year'].isna().any():
                    st.warning("Beberapa nilai di kolom 'Year' tidak dapat dikonversi ke format tanggal. Mencoba format lain...")
                    dataframe['Year'] = pd.to_datetime(dataframe['Year'], errors='coerce')
            
            result = arima_lstm_pred(dataframe)
            st.success(f"Prediksi inflasi bulan berikutnya: {result}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan upload file CSV yang berisi data inflasi historis untuk memulai prediksi.")
