# model.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from app.config import WINDOW_SIZE, PRED_DAYS

def prepare_data(df):
    close_prices = df['close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(len(scaled) - WINDOW_SIZE - PRED_DAYS + 1):
        X.append(scaled[i:i+WINDOW_SIZE, 0])
        y.append(scaled[i+WINDOW_SIZE:i+WINDOW_SIZE+PRED_DAYS, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # reshape for LSTM input
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(PRED_DAYS))
    model.compile(optimizer='adam', loss='mse')
    return model
