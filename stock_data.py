import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

class StockDataFetcher:
    def __init__(self, symbol='NVDA'):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)

    def get_realtime_data(self):
        """Get today's trading data"""
        try:
            # Get today's date in Eastern Time (ET)
            et_tz = pytz.timezone('US/Eastern')
            now = datetime.now(et_tz)
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            market_open = today.replace(hour=9, minute=30)  # Market opens at 9:30 AM ET

            # If it's before market open, use previous day's data
            if now < market_open:
                market_open = market_open - timedelta(days=1)

            data = self.stock.history(
                start=market_open,
                end=now,
                interval='1m'
            )
            return data
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_historical_data(self, days=30, interval='1d'):
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            data = self.stock.history(
                start=start_time,
                end=end_time,
                interval=interval
            )
            return data
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error