import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def get_data(ticker):

    df = yf.download(ticker, period="5y")

    data = df[["Close"]]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    # reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, data


def train_lstm(X, y):

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    return model