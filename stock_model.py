import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class StockPredictor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.accuracy = 0
        self.confidence = 0
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, data):
        # Add technical indicators
        data = data.copy()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])

        # Fill NaN values
        data = data.fillna(method='bfill')

        # Prepare features
        features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI']
        scaled_data = self.scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 0])  # Predict Close price

        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        return X, y

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def build_model(self, input_size):
        self.model = LSTMModel(input_size=input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train(self, data, epochs=100, validation_split=0.2, progress_bar=None):
        X, y = self.prepare_data(data)

        if self.model is None:
            self.build_model(X.shape[2])

        # Split data
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Train model
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training step
            self.model.train()  # Set to training mode
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            train_loss = self.criterion(outputs.squeeze(), y_train)
            train_loss.backward()
            self.optimizer.step()

            # Get training predictions before switching to eval mode
            train_preds = outputs.detach().squeeze().cpu().numpy()
            train_actual = y_train.cpu().numpy()
            train_acc = 1 - np.mean(np.abs((train_actual - train_preds) / train_actual))

            # Validation step
            self.model.eval()  # Set to evaluation mode
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs.squeeze(), y_val)
                val_preds = val_outputs.squeeze().cpu().numpy()
                val_actual = y_val.cpu().numpy()
                val_acc = 1 - np.mean(np.abs((val_actual - val_preds) / val_actual))

            # Store metrics
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # Update progress
            if progress_bar is not None:
                progress_bar.progress((epoch + 1) / epochs)
                progress_bar.text(f"""
                Epoch [{epoch+1}/{epochs}]
                Train Loss: {train_loss.item():.4f}
                Val Loss: {val_loss.item():.4f}
                Train Acc: {train_acc*100:.2f}%
                Val Acc: {val_acc*100:.2f}%
                """)

            # Print verbose output
            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"""
                Epoch [{epoch+1}/{epochs}]
                -------------------------
                Training Loss: {train_loss.item():.4f}
                Validation Loss: {val_loss.item():.4f}
                Training Accuracy: {train_acc*100:.2f}%
                Validation Accuracy: {val_acc*100:.2f}%
                Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}
                """)

            # Save best model
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                self.best_model_state = self.model.state_dict()

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            final_val_pred = self.model(X_val).squeeze().cpu().numpy()
            final_val_actual = y_val.cpu().numpy()

            self.accuracy = 1 - np.mean(np.abs((final_val_actual - final_val_pred) / final_val_actual))
            self.confidence = np.mean(np.exp(-np.abs(final_val_pred - final_val_actual)))

        self.is_trained = True
        return history

    def predict(self, data, days=7):
        if not self.is_trained:
            return None

        # Prepare last window of data
        data = data.copy()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data = data.fillna(method='bfill')

        features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI']
        scaled_data = self.scaler.transform(data[features].tail(self.window_size))

        # Make predictions
        predictions = []
        current_window = torch.FloatTensor(scaled_data).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(days):
                pred = self.model(current_window.unsqueeze(0)).item()
                predictions.append(pred)

                # Update window
                current_window = torch.roll(current_window, -1, dims=0)
                current_window[-1] = current_window[-2]  # Use last known values for features
                current_window[-1, 0] = pred  # Update predicted close price

        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, 0] = predictions
        unscaled_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]

        return unscaled_predictions