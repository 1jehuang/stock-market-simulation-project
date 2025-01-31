import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.accuracy = 0
        self.confidence = 0

    def prepare_data(self, data):
        # Prepare features (you can add more features here)
        features = ['Close', 'Volume']
        scaled_data = self.scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def build_model(self):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.window_size, 2)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, data, epochs=100):
        X, y = self.prepare_data(data)
        if self.model is None:
            self.build_model()

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train model
        self.model.fit(X_train, y_train,
                      epochs=epochs,
                      batch_size=32,
                      validation_data=(X_test, y_test),
                      verbose=0)

        # Calculate metrics
        pred = self.model.predict(X_test)
        self.accuracy = 1 - np.mean(np.abs(pred - y_test) / y_test)
        self.confidence = np.mean(np.exp(-np.abs(pred - y_test)))

    def predict(self, data, days=7):
        if self.model is None:
            return None

        # Prepare last window of data
        features = ['Close', 'Volume']
        scaled_data = self.scaler.transform(data[features].tail(self.window_size))

        # Make predictions
        predictions = []
        current_window = scaled_data.copy()

        for _ in range(days):
            pred = self.model.predict(current_window.reshape(1, self.window_size, 2))
            predictions.append(pred[0, 0])

            # Update window
            current_window = np.roll(current_window, -1, axis=0)
            current_window[-1, 0] = pred[0, 0]
            # Assume volume stays the same
            current_window[-1, 1] = current_window[-2, 1]

        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, 0] = predictions
        unscaled_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]

        return unscaled_predictions