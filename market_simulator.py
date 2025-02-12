import streamlit as st
import plotly.graph_objects as go
from stock_data import StockDataFetcher
import time
from trading import Portfolio
import pickle
import os
import pandas as pd
import numpy as np
from stock_model import StockPredictor

def load_or_create_model(symbol):
    model_path = f'models/{symbol}_model.pkl'
    if not os.path.exists('models'):
        os.makedirs('models')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return StockPredictor()

def main():
    # Load or create portfolio
    if os.path.exists('portfolio.pkl'):
        with open('portfolio.pkl', 'rb') as f:
            portfolio = pickle.load(f)
    else:
        portfolio = Portfolio()

    # Calculate total portfolio value at the start
    total_portfolio_value = portfolio.cash_balance
    for pos_symbol, position in portfolio.positions.items():
        fetcher = StockDataFetcher(pos_symbol)
        current_data = fetcher.get_realtime_data()
        if not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
            total_portfolio_value += position.shares * current_price

    # Remove default padding
    st.set_page_config(layout="wide")

    # Custom CSS to inject
    st.markdown("""
        <style>
        .main > div {
            padding-top: 0rem;
        }
        .stButton button {
            background-color: #1E1E1E;
            color: white;
            border: none;
            padding: 10px 24px;
        }
        .stMetric {
            background-color: #1E1E1E;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create the layout first
    left_col, main_col, model_col = st.columns([1, 3, 1])

    with left_col:  # Renamed from sidebar_col
        st.title('Trading Controls')

        # Now we can use total_portfolio_value here
        st.metric(
            label="Cash Balance",
            value=f"${portfolio.cash_balance:.2f}"
        )

        st.metric(
            label="Total Portfolio Value",
            value=f"${total_portfolio_value:.2f}"
        )

        st.markdown('---')

        # Trading controls
        shares = st.number_input('Number of Shares', min_value=1, value=1)
        order_type = st.selectbox('Order Type', ['Market', 'Limit'])

        if order_type == 'Limit':
            limit_price = st.number_input('Limit Price', min_value=0.01, format='%f')

        col1, col2 = st.columns(2)
        with col1:
            buy_button = st.button('Buy', use_container_width=True)
        with col2:
            sell_button = st.button('Sell', use_container_width=True)

        # Display portfolio positions
        st.markdown('---')
        st.subheader('Positions')
        for symbol, position in portfolio.positions.items():
            st.write(f"{symbol}: {position.shares} shares @ ${position.avg_price:.2f}")

    with main_col:
        # Use symbol from session state if available
        default_symbol = 'NVDA'  # Simplified to just use default
        symbol = st.text_input('Search for symbol', default_symbol,
                             help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")

        # Time interval buttons in a more compact layout
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            day = st.button('1D', use_container_width=True)
        with col2:
            week = st.button('1W', use_container_width=True)
        with col3:
            month = st.button('1M', use_container_width=True)
        with col4:
            three_month = st.button('3M', use_container_width=True)
        with col5:
            year = st.button('1Y', use_container_width=True)
        with col6:
            five_year = st.button('5Y', use_container_width=True)

        # Price metrics in a more compact layout
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price_container = st.empty()
        with col2:
            high_price_container = st.empty()
        with col3:
            low_price_container = st.empty()

        # Chart placeholder
        chart_placeholder = st.empty()

        # Initialize stock data fetcher
        fetcher = StockDataFetcher(symbol)

        # Determine selected time interval
        interval = '1d'  # default interval
        days = 1  # default days
        if day:
            interval, days = '1m', 1  # Changed to 1-minute intervals for more detail
        elif week:
            interval, days = '15m', 7
        elif month:
            interval, days = '1h', 30
        elif three_month:
            interval, days = '1d', 90
        elif year:
            interval, days = '1d', 365
        elif five_year:
            interval, days = '1d', 1825

        # Fetch data based on selected interval
        if days == 1:
            data = fetcher.get_realtime_data()
        else:
            data = fetcher.get_historical_data(days=days)

        # Check if we have data
        if data.empty:
            st.error(f"No data available for {symbol}. The market might be closed or the symbol might be invalid.")
            return

        # Handle trading actions
        try:
            current_price = data['Close'].iloc[-1]
        except (IndexError, KeyError):
            st.error("Unable to get current price. Please try again later.")
            return

        # After getting the data and current price, handle trading actions
        if buy_button:
            success, message = portfolio.place_order(
                symbol,
                shares,
                current_price,
                order_type
            )
            if success:
                st.sidebar.success(f"Bought {shares} shares of {symbol} at ${current_price:.2f}")
            else:
                st.sidebar.error(message)

        if sell_button:
            success, message = portfolio.place_order(
                symbol,
                -shares,  # Negative for selling
                current_price,
                order_type
            )
            if success:
                st.sidebar.success(f"Sold {shares} shares of {symbol} at ${current_price:.2f}")
            else:
                st.sidebar.error(message)

        # Save portfolio after any changes
        with open('portfolio.pkl', 'wb') as f:
            pickle.dump(portfolio, f)

        # Display portfolio positions with more detail
        st.sidebar.markdown('---')
        st.sidebar.subheader('Positions')

        if not portfolio.positions:
            st.sidebar.write("No open positions")
        else:
            for symbol, position in portfolio.positions.items():
                current_value = position.shares * current_price
                profit_loss = current_value - (position.shares * position.avg_price)
                profit_loss_pct = (profit_loss / (position.shares * position.avg_price)) * 100

                st.sidebar.markdown(f"""
                **{symbol}**
                - Shares: {position.shares}
                - Avg Price: ${position.avg_price:.2f}
                - Current Value: ${current_value:.2f}
                - P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)
                """)

        st.sidebar.markdown('---')
        # Calculate and display total portfolio value
        total_portfolio_value = portfolio.cash_balance
        for pos_symbol, position in portfolio.positions.items():
            if pos_symbol == symbol:  # If we have current price for this symbol
                total_portfolio_value += position.shares * current_price

        # Create and display chart
        fig = go.Figure()

        if not data.empty:  # Only create chart if we have data
            # Add line chart with enhanced hover template showing price change
            start_price = data['Close'].iloc[0]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                line=dict(color='#00FF1A', width=2),
                name='Price',
                hovertemplate="""
                <b>Price:</b> $%{y:.2f}<br>
                <b>Time:</b> %{x}<br>
                <b>Change:</b> %{customdata:.2f}%<br>
                <extra></extra>
                """,
                customdata=((data['Close'] - start_price) / start_price * 100),
                showlegend=False,
            ))

        # Calculate price range for y-axis
        price_min = data['Close'].min()
        price_max = data['Close'].max()
        price_range = price_max - price_min
        y_min = price_min - (price_range * 0.05)  # Add 5% padding
        y_max = price_max + (price_range * 0.05)

        # Update layout for dark theme with disabled zoom
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
                fixedrange=True,  # Disable y-axis zoom
            ),
            xaxis=dict(
                showgrid=False,
                showline=False,
                rangeslider=dict(visible=False),
                type='date',
                tickformat='%I:%M %p' if days == 1 else '%b %d' if days <= 90 else '%b %Y',
                tickfont=dict(color='white'),
                fixedrange=True,  # Disable x-axis zoom
            ),
            margin=dict(l=40, r=20, t=20, b=20),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='#2E2E2E',
                font_size=14,
                font_color='white',
                bordercolor='#3E3E3E'
            ),
            showlegend=False,
            height=700,  # Make the chart taller
            dragmode=False,  # Disable all dragging/zooming
        )

        # Display summary metrics
        current_price_container.metric(
            label="Current Price",
            value=f"${data['Close'].iloc[-1]:.2f}",
            delta=f"{((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100:.2f}%"
        )
        high_price_container.metric(
            label="High",
            value=f"${data['High'].max():.2f}"
        )
        low_price_container.metric(
            label="Low",
            value=f"${data['Low'].min():.2f}"
        )

        # Display the chart
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Now we can add the model visualization after we have the data
        with model_col:
            st.title("AI Model")

            # Model selection
            model_type = st.selectbox(
                "Select Model",
                ["LSTM", "Prophet", "Linear Regression", "Random Forest"]
            )

            if not data.empty:
                # Load or create model for current symbol
                model = load_or_create_model(symbol)

                # Model training status
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    st.warning("Model needs training")
                    train_needed = True
                else:
                    train_needed = False

                # Model status indicators
                st.markdown("### Model Status")
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    st.metric(
                        label="Accuracy",
                        value=f"{model.accuracy*100:.1f}%" if hasattr(model, 'accuracy') else "N/A",
                        delta=None
                    )
                with status_col2:
                    st.metric(
                        label="Confidence",
                        value=f"{model.confidence*100:.1f}%" if hasattr(model, 'confidence') else "N/A",
                        delta=None
                    )

                # Get predictions if model is trained
                if not train_needed:
                    predictions = model.predict(data)
                    if predictions is not None:
                        next_day = predictions[0]
                        week_ahead = predictions[-1]
                        current_price = data['Close'].iloc[-1]

                        # Calculate percentage changes
                        next_day_change = ((next_day - current_price) / current_price) * 100
                        week_change = ((week_ahead - current_price) / current_price) * 100

                        # Model prediction
                        st.markdown("### Prediction")
                        prediction_col1, prediction_col2 = st.columns(2)
                        with prediction_col1:
                            st.metric(
                                label="Next Day",
                                value=f"${next_day:.2f}",
                                delta=f"{next_day_change:.1f}%",
                                delta_color="normal" if next_day_change >= 0 else "inverse"
                            )
                        with prediction_col2:
                            st.metric(
                                label="7 Day",
                                value=f"${week_ahead:.2f}",
                                delta=f"{week_change:.1f}%",
                                delta_color="normal" if week_change >= 0 else "inverse"
                            )

                        # Model visualization
                        st.markdown("### Model Analysis")
                        prediction_fig = go.Figure()

                        # Add actual price line
                        prediction_fig.add_trace(go.Scatter(
                            x=data.index[-30:],
                            y=data['Close'].iloc[-30:],
                            mode='lines',
                            name='Actual',
                            line=dict(color='#00FF1A', width=2)
                        ))

                        # Add prediction line
                        future_dates = pd.date_range(start=data.index[-1], periods=len(predictions), freq='D')
                        prediction_fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines',
                            name='Predicted',
                            line=dict(color='#FF355E', width=2, dash='dash')
                        ))

                        prediction_fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='#1E1E1E',
                            paper_bgcolor='#1E1E1E',
                            height=200,
                            margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=True,
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                tickformat='$,.2f',
                            ),
                            xaxis=dict(
                                showgrid=False,
                            )
                        )

                        st.plotly_chart(prediction_fig, use_container_width=True)

                # Model parameters
                st.markdown("### Model Parameters")
                st.markdown(f"""
                - Window Size: {model.window_size} days
                - Learning Rate: 0.001
                - Epochs: 100
                - Features: Price, Volume
                """)

                # Training button
                if st.button("Train Model", use_container_width=True):
                    # Create placeholders for progress and metrics
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_cols = st.columns(2)

                    train_metric = metrics_cols[0].empty()
                    val_metric = metrics_cols[1].empty()

                    # Create a plot placeholder for loss curves
                    loss_plot = st.empty()

                    with st.spinner('Training model...'):
                        try:
                            # Train the model with progress tracking
                            history = model.train(data, progress_bar=progress_bar)

                            # Create loss curves plot
                            fig = go.Figure()
                            epochs = list(range(1, len(history['train_loss']) + 1))

                            # Add training loss
                            fig.add_trace(go.Scatter(
                                x=epochs,
                                y=history['train_loss'],
                                mode='lines',
                                name='Training Loss',
                                line=dict(color='#00FF1A')
                            ))

                            # Add validation loss
                            fig.add_trace(go.Scatter(
                                x=epochs,
                                y=history['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#FF355E')
                            ))

                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='#1E1E1E',
                                paper_bgcolor='#1E1E1E',
                                title='Training Progress',
                                xaxis_title='Epoch',
                                yaxis_title='Loss',
                                height=300
                            )

                            # Update the plot
                            loss_plot.plotly_chart(fig, use_container_width=True)

                            # Save the trained model
                            with open(f'models/{symbol}_model.pkl', 'wb') as f:
                                pickle.dump(model, f)

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.success('Model trained successfully!')

                            # Display final metrics
                            train_metric.metric(
                                "Final Training Accuracy",
                                f"{history['train_acc'][-1]*100:.2f}%"
                            )
                            val_metric.metric(
                                "Final Validation Accuracy",
                                f"{history['val_acc'][-1]*100:.2f}%"
                            )

                        except Exception as e:
                            st.error(f'Error training model: {str(e)}')
                            print(f"Detailed error: {e}")  # For debugging
            else:
                st.write("No data available for model analysis")

        # Refresh every minute
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()