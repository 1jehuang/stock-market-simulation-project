# Stock Market Simulator with AI Predictions

A simulation platform that combines real-time trading with AI-driven price predictions. Users can trade in a risk-free environment while assessing the accuracy of machine learning insights.

## Features
- **Real-time Market Tracking** – Follow live stock prices and trends.
- **AI-Powered Predictions** – LSTM, Prophet, Linear Regression, and Random Forest models predict short- and medium-term price movements with confidence metrics.
- **Realistic Trading Interface** – Market/limit orders, portfolio tracking, P&L calculations, and position management.
- **Custom Model Integration** – Easily add and test your own prediction models.

## Adding Custom Models
1. Create a new class in `stock_model.py`
2. Implement `train` and `predict` methods
3. Register your model in the UI for testing

## Usage Instructions

### Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. ```bash
    streamlit run market_simulator.py