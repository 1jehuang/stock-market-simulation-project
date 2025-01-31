import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from stock_data import StockDataFetcher
from trading import Portfolio
import pickle
import os

def load_portfolio():
    if os.path.exists('portfolio.pkl'):
        with open('portfolio.pkl', 'rb') as f:
            return pickle.load(f)
    return Portfolio()

def get_portfolio_history(portfolio, days=30):
    """Calculate historical portfolio value"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Create date range for the portfolio history
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    portfolio_values = []

    # Calculate portfolio value for each date
    for date in dates:
        daily_value = portfolio.cash_balance  # Start with cash balance

        # Add value of each position
        for symbol, position in portfolio.positions.items():
            fetcher = StockDataFetcher(symbol)
            # Get historical data up to this date
            hist_data = fetcher.get_historical_data(
                days=(end_date - date).days + 1
            )

            if not hist_data.empty:
                # Get the closing price closest to this date
                mask = hist_data.index.date <= date.date()
                if mask.any():
                    price = hist_data[mask]['Close'].iloc[-1]
                    daily_value += position.shares * price

        portfolio_values.append(daily_value)

    # Create DataFrame with dates and values
    portfolio_history = pd.DataFrame({
        'value': portfolio_values
    }, index=dates)

    return portfolio_history

def display_portfolio_chart(portfolio_history):
    """Create a chart for portfolio value"""
    fig = go.Figure()

    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=portfolio_history.index,
        y=portfolio_history['value'],
        mode='lines',
        line=dict(color='#00FF1A', width=2),
        hovertemplate="""
        <b>Value:</b> $%{y:.2f}<br>
        <b>Date:</b> %{x}<br>
        <extra></extra>
        """,
        showlegend=False,
    ))

    # Calculate min/max for y-axis padding
    y_min = portfolio_history['value'].min()
    y_max = portfolio_history['value'].max()
    y_range = y_max - y_min
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=False,
            tickformat='$,.2f',
            range=[y_min, y_max],
            tickmode='auto',
            nticks=8,
            tickfont=dict(color='white'),
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            type='date',
            tickfont=dict(color='white'),
            fixedrange=True,
        ),
        margin=dict(l=40, r=20, t=20, b=20),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#2E2E2E',
            font_size=14,
            font_color='white',
            bordercolor='#3E3E3E'
        ),
        height=400,
        dragmode=False,
    )

    return fig

def main():
    st.set_page_config(layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .main > div {
            padding-top: 0rem;
        }
        .stock-card {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stock-card:hover {
            background-color: #2E2E2E;
        }
        </style>
    """, unsafe_allow_html=True)

    portfolio = load_portfolio()

    # Portfolio Chart Section
    st.subheader("Portfolio Value")

    # Time interval buttons for portfolio chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        portfolio_1m = st.button('1M', key='portfolio_1m')
    with col2:
        portfolio_3m = st.button('3M', key='portfolio_3m')
    with col3:
        portfolio_6m = st.button('6M', key='portfolio_6m')
    with col4:
        portfolio_1y = st.button('1Y', key='portfolio_1y')

    # Update portfolio history based on selected timeframe
    days = 30  # default to 1 month
    if portfolio_3m:
        days = 90
    elif portfolio_6m:
        days = 180
    elif portfolio_1y:
        days = 365

    # Get and display portfolio history
    portfolio_history = get_portfolio_history(portfolio, days=days)

    # Add percentage change calculation
    if not portfolio_history.empty:
        start_value = portfolio_history['value'].iloc[0]
        end_value = portfolio_history['value'].iloc[-1]
        pct_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

        # Display total value with percentage change
        st.metric(
            label="Portfolio Value",
            value=f"${end_value:.2f}",
            delta=f"{pct_change:.2f}%"
        )

        # Display the chart
        fig = display_portfolio_chart(portfolio_history)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Start trading to see your portfolio value chart!")

    # Portfolio Overview Section
    col1, col2, col3 = st.columns(3)

    # Calculate total value including all positions
    total_value = portfolio.cash_balance
    for symbol, position in portfolio.positions.items():
        fetcher = StockDataFetcher(symbol)
        current_data = fetcher.get_realtime_data()
        if not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
            total_value += position.shares * current_price

    with col1:
        st.metric(
            label="Total Portfolio Value",
            value=f"${total_value:.2f}",
            delta=f"{((total_value - portfolio.cash_balance) / portfolio.cash_balance * 100):.2f}%" if portfolio.cash_balance > 0 else "0.00%"
        )

    with col2:
        st.metric(
            label="Buying Power",
            value=f"${portfolio.cash_balance:.2f}"
        )

    with col3:
        invested_value = total_value - portfolio.cash_balance
        st.metric(
            label="Total Invested",
            value=f"${invested_value:.2f}"
        )

    # Stock Positions Section
    st.markdown("---")
    st.subheader("Your Positions")

    if not portfolio.positions:
        st.write("No positions yet. Start trading to build your portfolio!")
    else:
        # Create a grid layout for position cards
        for symbol, position in portfolio.positions.items():
            fetcher = StockDataFetcher(symbol)
            current_data = fetcher.get_realtime_data()

            if not current_data.empty:
                current_price = current_data['Close'].iloc[-1]
                position_value = position.shares * current_price
                profit_loss = position_value - (position.shares * position.avg_price)
                profit_loss_pct = (profit_loss / (position.shares * position.avg_price)) * 100

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="stock-card">
                        <h3>{symbol}</h3>
                        <p>Shares: {position.shares}</p>
                        <p>Current Value: ${position_value:.2f}</p>
                        <p>P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button('Trade', key=f"trade_{symbol}"):
                        st.session_state.page = "trade"
                        st.session_state.symbol = symbol
                        st.rerun()

    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Trade New Stock"):
        st.session_state.page = "trade"
        st.rerun()

if __name__ == "__main__":
    main()