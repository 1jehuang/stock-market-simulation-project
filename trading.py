from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class Position:
    symbol: str
    shares: int
    avg_price: float

class Portfolio:
    def __init__(self, initial_balance=10000):
        self.cash_balance = initial_balance
        self.positions = {}  # symbol -> Position
        self.trade_history = []

    def place_order(self, symbol: str, shares: int, price: float, order_type: str):
        """
        Place a buy or sell order
        shares: positive for buy, negative for sell
        """
        total_cost = shares * price

        # Check if selling
        if shares < 0:
            if symbol not in self.positions or self.positions[symbol].shares < abs(shares):
                return False, "Not enough shares to sell"

        # Check if buying
        if shares > 0:
            if total_cost > self.cash_balance:
                return False, "Insufficient funds"

        # Execute trade
        self.cash_balance -= total_cost

        if symbol in self.positions:
            current_position = self.positions[symbol]
            new_shares = current_position.shares + shares

            if new_shares == 0:
                del self.positions[symbol]
            else:
                if shares > 0:  # Buying more
                    avg_price = ((current_position.avg_price * current_position.shares) +
                               (price * shares)) / new_shares
                else:  # Selling some
                    avg_price = current_position.avg_price

                self.positions[symbol] = Position(symbol, new_shares, avg_price)
        else:
            self.positions[symbol] = Position(symbol, shares, price)

        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'type': 'buy' if shares > 0 else 'sell',
            'total': total_cost
        })

        return True, "Trade executed successfully"

    def get_position_value(self, symbol: str, current_price: float) -> float:
        if symbol in self.positions:
            return self.positions[symbol].shares * current_price
        return 0.0

    def get_position_summary(self, symbol: str, current_price: float) -> dict:
        """Get detailed summary of a position"""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        current_value = position.shares * current_price
        cost_basis = position.shares * position.avg_price
        profit_loss = current_value - cost_basis
        profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis != 0 else 0

        return {
            'shares': position.shares,
            'avg_price': position.avg_price,
            'current_value': current_value,
            'cost_basis': cost_basis,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        }

    def get_total_value(self, current_prices: dict) -> float:
        """Calculate total portfolio value including cash"""
        total = self.cash_balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position.shares * current_prices[symbol]
        return total