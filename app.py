import streamlit as st
from streamlit_option_menu import option_menu
from pygwalker.api.streamlit import StreamlitRenderer
from streamlit_lottie import st_lottie
import json
import itertools
import pandas as pd
from datetime import datetime
import numpy as np
import time
from arch import arch_model
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load the Lottie animation
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the animation file
lottie_animation = load_lottie_file("model_loading_animation.json")

# Set up the navigation bar in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Data Preprocessing & Exploration", "EMA and GJR-GARCH(1,1)"],
        icons=["table", "graph-up-arrow", "graph-down", "bar-chart", "clock"],
        menu_icon="cast",
        default_index=0
    )

# Data Preprocessing & Exploration Section
if selected == "Data Preprocessing & Exploration":
    st.title("Data Preprocessing and Exploration")
    st.write("Select an expiry date to view the close prices for the corresponding contract.")

    # Load and preprocess data
    future_df = pd.read_csv("Future_CRUDE_2005_2023.csv")
    future_df['Date'] = pd.to_datetime(future_df['Date'])
    future_df = future_df.sort_values(by=['Expiry Date', 'Date'])
    future_df = future_df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
    future_df = future_df[['Date', 'Expiry Date', 'Close']]
    future_df.dropna(inplace=True)
    future_df = future_df.reset_index(drop=True)
    future_df['Log Return'] = np.log(future_df.groupby('Expiry Date')['Close'].shift(-1) / future_df['Close'])
    future_df.dropna(subset=['Log Return'], inplace=True)
    future_df = future_df[-850:]  # Use the most recent 850 rows

    # Dropdown for selecting an expiry date
    unique_expiry_dates = future_df['Expiry Date'].unique()
    selected_expiry_date = st.selectbox("Select Expiry Date", unique_expiry_dates)

    # Filter data for the selected expiry date
    filtered_df = future_df[future_df['Expiry Date'] == selected_expiry_date]
    dates = filtered_df['Date'].dt.strftime('%Y-%m-%d').tolist()
    close_prices = filtered_df['Close'].tolist()

    # Close Price Line Graph for Selected Contract
    chart_option = {
        "title": {"text": f"Close Prices for Expiry Date {selected_expiry_date}"},
        "tooltip": {
            "trigger": "axis",  # Show tooltip when hovering over axis points
            "formatter": "{b}: {c}",  # '{b}' will be the X value (Date), '{c}' will be the Close Price
        },
        "xAxis": {"type": "category", "data": dates},
        "yAxis": {"type": "value", "name": "Close Price"},
        "series": [{
            "data": close_prices,
            "type": "line",
            "smooth": True
        }]
    }

    # Display the chart with tooltips
    st_echarts(options=chart_option, height="400px")

    # Graph: Close Price Movement Throughout the Dataset
    st.subheader("Close Price Movement Throughout the Dataset")
    all_dates = future_df['Date'].dt.strftime('%Y-%m-%d').tolist()
    all_close_prices = future_df['Close'].tolist()
    all_close_option = {
        "title": {"text": "Close Prices Throughout the Dataset"},
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}"},
        "xAxis": {"type": "category", "data": all_dates},
        "yAxis": {"type": "value", "name": "Close Price"},
        "series": [{
            "data": all_close_prices,
            "type": "line",
            "smooth": True
        }]
    }
    st_echarts(options=all_close_option, height="400px")

    # Graph: Log Return Movement Throughout the Dataset
    st.subheader("Log Return Movement Throughout the Dataset")
    fig, ax = plt.subplots()
    ax.plot(future_df['Date'], future_df['Log Return'], color="orange")
    ax.set_title("Log Return Movement")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.grid(True)
    st.pyplot(fig)

    # Log Return Distribution for All Data
    st.subheader("Log Return Distribution")
    log_returns = future_df['Log Return'].dropna()

    # Calculate mean and std_dev using the cleaned log returns
    mean, std_dev = log_returns.mean(), log_returns.std()

    # Histogram of Log Returns with Normal Distribution Overlay
    fig, ax = plt.subplots()
    ax.hist(log_returns, bins=20, density=True, alpha=0.6, color="skyblue", edgecolor="black")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    ax.plot(x, p, "r", linewidth=2)
    ax.set_title("Log Return Histogram")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    st.pyplot(fig)

    # ACF and PACF Plots for Log Returns
    st.subheader("ACF and PACF Plots for Log Returns")

    # ACF Plot
    fig_acf, ax_acf = plt.subplots()
    plot_acf(log_returns, ax=ax_acf, lags=20, alpha=0.05)
    ax_acf.set_title("Autocorrelation Function (ACF)")
    st.pyplot(fig_acf)

    # PACF Plot
    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(log_returns, ax=ax_pacf, lags=20, alpha=0.05, method='ywm')
    ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig_pacf)

    # Ljung-Box Test for Autocorrelation
    st.subheader("Ljung-Box Test for Autocorrelation")
    lb_test = acorr_ljungbox(log_returns, lags=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], return_df=True)
    st.write("Ljung-Box Test Results at 20 lags:")
    st.write(lb_test)


# Strategy Simulation
if selected == "EMA and GJR-GARCH(1,1)":
    future_df = pd.read_csv("Future_CRUDE_2005_2023.csv")
    future_df['Date'] = pd.to_datetime(future_df['Date'])
    future_df = future_df.sort_values(by=['Expiry Date', 'Date'])
    future_df = future_df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
    future_df = future_df[['Date', 'Expiry Date', 'Close']]
    future_df.dropna(inplace=True)
    future_df = future_df.reset_index(drop=True)
    # Convert 'Date' to datetime for plotting purposes
    future_df['Date'] = pd.to_datetime(future_df['Date'])

    # Calculate Log Return as ln(Close of next row / Close of current row) within each Expiry Date
    future_df['Log Return'] = np.log(future_df.groupby('Expiry Date')['Close'].shift(-1) / future_df['Close'])

    # Drop any rows with NaN values in the Log Return column (last row per group will have NaN Log Return)
    future_df.dropna(subset=['Log Return'], inplace=True)
    st.title("EMA and GJR-GARCH(1,1) Trading Strategy Simulation")
    st.write("Select parameters to run simulations for the EMA and GJR-GARCH(1,1) trading strategies.")
    
    st.header("Strategy 1: EMA and GJR-GARCH(1,1) Overview")
    st.write("""
    This strategy combines Exponential Moving Average (EMA) with a GJR-GARCH(1,1) model to capture market trends and volatility patterns.
    The EMA is used to establish a drift in asset prices, while the GJR-GARCH(1,1) model forecasts volatility. The simulation predicts
    future price paths over a short horizon - time till expiry (e.g., 10 days). Based on the probability of reaching a 3% target increase, a BUY signal
    is triggered, or if a 3% target decrease is likely, a SELL signal is generated.
    """)


    
    # Simulation Parameters
    num_days = 10
    num_simulations = 1000
    initial_price = 100
    target_increase = 0.03  # 3% increase for BUY condition
    target_decrease = -0.03  # 3% decrease for SELL condition

    # Function to simulate price paths
    def simulate_price_paths(initial_price, drift, volatility, num_days, num_simulations):
        price_paths = np.zeros((num_days, num_simulations))
        price_paths[0] = initial_price
        for t in range(1, num_days):
            Z = np.random.standard_normal(num_simulations)
            price_paths[t] = price_paths[t-1] * np.exp((drift - 0.5 * volatility**2) + volatility * Z)
        return price_paths

    # Example drift and volatility
    drift = 0.005  # Sample drift based on EMA
    volatility = 0.02  # Sample volatility from GJR-GARCH(1,1)

    # Simulate price paths
    buy_paths = simulate_price_paths(initial_price, drift, volatility, num_days, num_simulations)
    
    drift = -0.005
    sell_paths = simulate_price_paths(initial_price, drift, volatility, num_days, num_simulations)

    # Calculate probabilities of hitting 3% targets
    buy_target = initial_price * (1 + target_increase)
    sell_target = initial_price * (1 + target_decrease)
    prob_buy = (buy_paths[-1] >= buy_target).mean()
    prob_sell = (sell_paths[-1] <= sell_target).mean()

    # Plot BUY condition
    fig, ax = plt.subplots()
    ax.plot(buy_paths[:, :1000], color='blue', alpha=0.3)  # Show a few simulation paths
    ax.axhline(buy_target, color='red', linestyle='--', label="3% Above Target")
    ax.plot(0, initial_price, 'ro')  # Starting price red dot
    ax.set_title(f"BUY Condition - Probability of 3% Higher: {prob_buy:.2%}")
    ax.legend()
    st.pyplot(fig)

    # Plot SELL condition
    fig, ax = plt.subplots()
    ax.plot(sell_paths[:, :1000], color='purple', alpha=0.3)  # Show a few simulation paths
    ax.axhline(sell_target, color='red', linestyle='--', label="3% Below Target")
    ax.plot(0, initial_price, 'ro')  # Starting price red dot
    ax.set_title(f"SELL Condition - Probability of 3% Lower: {prob_sell:.2%}")
    ax.legend()
    st.pyplot(fig)
    
    st.header("Strategy 2: EMA and GJR-GARCH(1,1) Overview")
    st.write("""This strategy trades based on simulated price movements, entering **Buy** when the price is likely to rise and **Sell** when the price is likely to fall, based on probability thresholds. It allows a maximum of two open positions at a time. Positions are closed when the **profit target** or **stop loss** is reached, with **Buy** positions exited by **Sell** and vice versa. Entries and exits are plotted, ensuring clear trade management while controlling risk.""")
    
        # Parameters
    forecast_horizon = 20          # Forecast horizon in time steps
    num_simulations = 1000         # Number of simulated paths
    drift = 0.0002                 # Drift for log returns
    volatility = 0.02              # Volatility for log returns
    profit_target = 0.03           # 3% profit target
    stop_loss = -0.02              # 2% stop-loss
    entry_threshold_prob = 0.6     # Entry threshold probability
    max_open_positions = 2         # Maximum number of open positions allowed

    # Generate synthetic actual price data for demonstration
    np.random.seed(64)
    actual_price = 100             # Starting actual price
    actual_prices = [actual_price]
    for _ in range(forecast_horizon):
        actual_price *= np.exp(drift + volatility * np.random.normal())
        actual_prices.append(actual_price)
    actual_prices = np.array(actual_prices)

    # Simulate price paths
    simulated_prices = np.zeros((forecast_horizon + 1, num_simulations))
    simulated_prices[0] = actual_prices[0]
    for t in range(1, forecast_horizon + 1):
        Z = np.random.normal(size=num_simulations)
        simulated_prices[t] = simulated_prices[t - 1] * np.exp(drift - 0.5 * volatility**2 + volatility * Z)

    # Initialize trade records
    entry_points = {'buy': [], 'sell': []}
    exit_points = {'buy': [], 'sell': []}

    # Track open positions
    open_positions = []  # List of open positions

    # Simulate entry and exit criteria over forecast horizon
    logs = []  # To store print logs

    for t in range(1, forecast_horizon + 1):
        # Calculate probabilities of the simulated price being above or below the actual price
        above_prob = (simulated_prices[t] > actual_prices[t]).mean()
        below_prob = (simulated_prices[t] < actual_prices[t]).mean()
        
        # Entry conditions: open a new position if probability threshold is met and positions < max_open_positions
        if len(open_positions) < max_open_positions:
            if above_prob > entry_threshold_prob:
                open_positions.append({"type": "buy", "price": actual_prices[t], "time": t})
                entry_points['buy'].append((t, actual_prices[t]))
                logs.append(f"Entry: Buy at time {t}, price: {actual_prices[t]:.2f} - Reason: above_prob {above_prob:.2f} > threshold {entry_threshold_prob}")
            elif below_prob > entry_threshold_prob:
                open_positions.append({"type": "sell", "price": actual_prices[t], "time": t})
                entry_points['sell'].append((t, actual_prices[t]))
                logs.append(f"Entry: Sell at time {t}, price: {actual_prices[t]:.2f} - Reason: below_prob {below_prob:.2f} > threshold {entry_threshold_prob}")
        
        # Exit conditions: check each open position and exit if profit target or stop loss is reached
        closed_positions = []
        for position in open_positions:
            entry_price = position['price']
            position_type = position['type']
            
            # Calculate return for position
            trade_return = (actual_prices[t] - entry_price) / entry_price if position_type == "buy" else (entry_price - actual_prices[t]) / entry_price
            
            # Exit conditions based on profit target or stop loss
            if position_type == "buy" and (trade_return >= profit_target or trade_return <= stop_loss):
                # Exit Buy position by opening Sell position (close Buy)
                exit_points['sell'].append((t, actual_prices[t]))  # Mark exit as Sell to close Buy
                logs.append(f"Exit Sell at time {t}, price: {actual_prices[t]:.2f} - Reason: {'profit target' if trade_return >= profit_target else 'stop loss'} reached (return {trade_return:.2f})")
                closed_positions.append(position)
            elif position_type == "sell" and (trade_return >= profit_target or trade_return <= stop_loss):
                # Exit Sell position by opening Buy position (close Sell)
                exit_points['buy'].append((t, actual_prices[t]))  # Mark exit as Buy to close Sell
                logs.append(f"Exit Buy at time {t}, price: {actual_prices[t]:.2f} - Reason: {'profit target' if trade_return >= profit_target else 'stop loss'} reached (return {trade_return:.2f})")
                closed_positions.append(position)

        # Remove closed positions from open positions
        open_positions = [pos for pos in open_positions if pos not in closed_positions]

    # Display the logs as a numbered list in Streamlit
    st.write("### Logs:")
    for idx, log in enumerate(logs, 1):
        st.write(f"{idx}. {log}")

    # Mark entry and exit points on the plot with different symbols for Buy and Sell
    plt.figure(figsize=(14, 7))
    for i in range(num_simulations):
        plt.plot(simulated_prices[:, i], color="grey", alpha=0.2)
    plt.plot(actual_prices, color="blue", label="Actual Price", linewidth=2)

    # Plot entry and exit points after the simulated paths to ensure they are on top
    for entry in entry_points['buy']:
        plt.scatter(entry[0], entry[1], color="green", marker="^", s=100, label="Entry Buy" if entry == entry_points['buy'][0] else "",zorder=3)
    for entry in entry_points['sell']:
        plt.scatter(entry[0], entry[1], color="purple", marker="v", s=100, label="Entry Sell" if entry == entry_points['sell'][0] else "",zorder=3)
    for exit in exit_points['buy']:
        plt.scatter(exit[0], exit[1], color="red", marker="x", s=100, label="Exit Buy" if exit == exit_points['buy'][0] else "",zorder=3)
    for exit in exit_points['sell']:
        plt.scatter(exit[0], exit[1], color="orange", marker="o", s=100, label="Exit Sell" if exit == exit_points['sell'][0] else "",zorder=3)

    # Customize the plot
    plt.title("Simulated Price Paths with Entry and Exit Points")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)
    
    # User inputs for date range and trading parameters
    start_date = st.date_input("Start Date", datetime(2022, 8, 19))
    end_date = st.date_input("End Date", datetime(2023, 10, 19))
    
    # Choose trading strategy
    strategy = st.selectbox("Choose a Trading Strategy", ["Strategy 1", "Strategy 2"])

    # Strategy parameters
    if strategy=="Strategy 1":
        max_positions = st.slider("Max Positions", 1, 5, 2)
        profit_target = st.slider("Profit Target (%)", 0.01, 0.1, 0.03)
        stop_loss = st.slider("Stop Loss (%)", -0.1, 0.0, -0.02)
    else:
        max_positions = st.slider("Max Positions", 1, 5, 2)
        profit_target = st.slider("Profit Target (%)", 0.01, 0.1, 0.03)
        stop_loss = st.slider("Stop Loss (%)", -0.1, 0.0, -0.02)
        initial_threshold_prob = st.slider("Initial Threshold Probability", 0.5, 0.9, 0.6)
        initial_decline_threshold = st.slider("Initial Decline Threshold", 0.01, 0.2, 0.06)

    # Set up date ranges for test indices
    test_dates = [
        '2022-08-19', '2022-09-19', '2022-10-19', '2022-11-18', '2022-12-19',
        '2023-01-19', '2023-02-17', '2023-03-20', '2023-04-19', '2023-05-19',
        '2023-06-16', '2023-07-19', '2023-08-21', '2023-09-19', '2023-10-19'
    ]
    test_indices = [[-307,-286],[-286,-266],[-266,-245],[-245,-224],[-224,-204],
                    [-204,-182],[-182,-163],[-163,-143],[-143,-123],[-123,-102],
                    [-102,-83],[-83,-61],[-61,-40],[-40,-20],[-20,0]]

    # Filter test_indices based on selected date range
    filtered_indices = [
        test for date, test in zip(test_dates, test_indices)
        if start_date <= datetime.strptime(date, '%Y-%m-%d').date() <= end_date
    ]
    

    # Strategy 1
    def strategy_1(future_df, filtered_indices, num_runs=100):
        # Initialize lists to store performance metrics across multiple runs
        accuracy_list = []
        accumulated_return_list = []
        average_duration_list = []
        average_total_trades = []
        average_total_return = []
        average_total_principal = []

        # Define window size for rolling GARCH model
        window_size = 250  # Size of the window for rolling estimation

        for _ in range(num_runs):
            # Initialize performance tracking variables for each run
            total_trades = 0
            successful_trades = 0
            accumulated_return = 0
            durations_list = []
            total_return = 0
            total_principal = 0


            # Calculate EMA of Log Returns for drift estimation (span 25)
            future_df['EMA Drift'] = future_df['Log Return'].ewm(span=25, adjust=True).mean()

            for test in filtered_indices:
                forecast_horizon = test[1] - test[0]
                actual_prices = future_df['Close'][test[0]:test[1]].values if test[1] != 0 else future_df['Close'][test[0]:].values

                # Step 2: Fit a GJR-GARCH(1,1) model on a rolling window
                if test[0] >= window_size:
                    garch_model = arch_model(future_df['Log Return'][test[0]-window_size:test[0]], vol='Garch', p=1, q=1, o=1)
                else:
                    garch_model = arch_model(future_df['Log Return'][:test[0]], vol='Garch', p=1, q=1, o=1)

                garch_results = garch_model.fit(disp="off")
                garch_forecast = garch_results.forecast(horizon=forecast_horizon)
                volatility_forecast = garch_forecast.variance.values[-1, :] ** 0.5

                # Step 3: Strategy simulation across forecast horizon
                last_price = future_df['Close'].iloc[test[0]]
                open_positions = []

                for t in range(forecast_horizon):
                    # Calculate days left to expiry and dynamic simulation horizon
                    days_left_to_expiry = forecast_horizon - t
                    simulation_days = max(1, days_left_to_expiry)

                    # Get the latest EMA Drift and volatility for the current step
                    current_drift = future_df['EMA Drift'].iloc[test[0] + t]
                    current_volatility = volatility_forecast[t]

                    # Simulate future price paths
                    num_simulations = 10000
                    simulated_prices = np.zeros((simulation_days, num_simulations))
                    simulated_prices[0] = actual_prices[t]

                    for sim_t in range(1, simulation_days):
                        Z = np.random.standard_normal(num_simulations)
                        drift = current_drift - 0.5 * current_volatility**2
                        simulated_prices[sim_t] = simulated_prices[sim_t-1] * np.exp(drift + current_volatility * Z)

                    # Calculate probabilities for price movements
                    target_price_up = actual_prices[t] * 1.03
                    target_price_down = actual_prices[t] * 0.97
                    above_prob = (simulated_prices[-1] >= target_price_up).mean()
                    below_prob = (simulated_prices[-1] <= target_price_down).mean()

                    # Generate buy/sell signals based on probability thresholds
                    if len(open_positions) < max_positions:
                        if above_prob > 0.5:
                            open_positions.append((actual_prices[t], 1, t))
                        elif below_prob > 0.5:
                            open_positions.append((actual_prices[t], -1, t))

                    # Check for exit criteria for open positions
                    closed_positions = []
                    for position in open_positions:
                        entry_price, position_type, entry_time = position
                        trade_return = ((actual_prices[t] - entry_price) / entry_price if position_type == 1
                                        else (entry_price - actual_prices[t]) / entry_price)


                        # Exit if conditions for profit target, stop loss, or horizon expiry are met
                        if ((position_type == 1 and (trade_return >= profit_target or trade_return <= stop_loss)) or
                            (position_type == -1 and (trade_return >= profit_target or trade_return <= stop_loss))):
                            total_trades += 1
                            accumulated_return += trade_return
                            total_return += ((actual_prices[t] - entry_price)  if position_type == 1
                                        else (entry_price - actual_prices[t]) )
                            average_total_principal.append(entry_price)
                            durations_list.append(t - entry_time)
                            if trade_return > 0:
                                successful_trades += 1
                            closed_positions.append(position)

                    # Remove closed positions from the open positions list
                    open_positions = [p for p in open_positions if p not in closed_positions]

                # Close remaining open positions at the end of the horizon
                final_price = actual_prices[-1]
                for position in open_positions:
                    entry_price, position_type, entry_time = position
                    trade_return = ((final_price - entry_price) / entry_price if position_type == 1
                                    else (entry_price - final_price) / entry_price)
                    total_trades += 1
                    accumulated_return += trade_return
                    total_return += ((actual_prices[t] - entry_price)  if position_type == 1
                                        else (entry_price - actual_prices[t]) )
                    average_total_principal.append(entry_price)
                    durations_list.append(forecast_horizon - entry_time)
                    if trade_return > 0:
                        successful_trades += 1

            # Record performance metrics for each run
            accuracy = successful_trades / total_trades if total_trades > 0 else 0
            average_duration = np.mean(durations_list) if durations_list else 0
            accuracy_list.append(accuracy)
            accumulated_return_list.append(accumulated_return)
            average_duration_list.append(average_duration)
            average_total_trades.append(total_trades)
            average_total_return.append(total_return)
        
        return np.mean(accuracy_list), np.mean(accumulated_return_list), np.mean(average_total_trades),np.mean(average_duration_list),np.mean(average_total_return),np.mean(average_total_principal),np.mean(average_total_return)/(np.mean(average_total_principal)*max_positions*0.3)

    # Strategy 2 (Modify accordingly)
    def strategy_2(future_df, filtered_indices, num_runs=100):
        # Initialize lists to store performance metrics across multiple runs
        accuracy_list = []
        accumulated_return_list = []
        average_duration_list = []
        average_total_trades = []
        average_total_return = []
        average_total_principal = []
        num_runs = 100

        # Define window size for rolling GARCH model
        window_size = 250  # Size of the window for rolling estimation

        for _ in range(num_runs):
            # Initialize performance tracking variables for each run
            total_trades = 0
            successful_trades = 0
            accumulated_return = 0
            durations_list = []
            total_return = 0
            total_principal = 0

            # Calculate EMA of Log Returns for drift estimation (span 25)
            future_df['EMA Drift'] = future_df['Log Return'].ewm(span=25, adjust=True).mean()

            for test in filtered_indices:
                forecast_horizon = test[1] - test[0]
                actual_prices = future_df['Close'][test[0]:test[1]].values if test[1] != 0 else future_df['Close'][test[0]:].values

                # Step 2: Fit a GJR-GARCH(1,1) model on a rolling window
                if test[0] >= window_size:
                    garch_model = arch_model(future_df['Log Return'][test[0]-window_size:test[0]], vol='Garch', p=1, q=1, o=1)  # GJR-GARCH(1,1)
                else:
                    garch_model = arch_model(future_df['Log Return'][:test[0]], vol='Garch', p=1, q=1, o=1)  # GJR-GARCH(1,1)

                garch_results = garch_model.fit(disp="off")
                garch_forecast = garch_results.forecast(horizon=forecast_horizon)
                volatility_forecast = garch_forecast.variance.values[-1, :] ** 0.5

                # Retrieve the EMA drift specific to the current point
                ema_drift = future_df['EMA Drift'].iloc[test[0]]

                # Step 3: Simulate future price paths with GARCH volatility and EMA drift
                num_simulations = 1000
                last_price = future_df['Close'].iloc[test[0]]
                simulated_prices = np.zeros((forecast_horizon, num_simulations))
                simulated_prices[0] = last_price

                for t in range(1, forecast_horizon):
                    Z = np.random.standard_normal(num_simulations)
                    drift = ema_drift - 0.5 * volatility_forecast[t]**2
                    simulated_prices[t] = simulated_prices[t-1] * np.exp(drift + volatility_forecast[t] * Z)

                # Initialize lists to store probabilities and positions
                above_actual = []
                below_actual = []
                open_positions = []

                for t in range(forecast_horizon):
                    decayed_threshold_prob = initial_threshold_prob
                    decayed_decline_threshold = initial_decline_threshold
                    above_prob = (simulated_prices[t] > actual_prices[t]).mean()
                    below_prob = (simulated_prices[t] < actual_prices[t]).mean()
                    above_actual.append(above_prob)
                    below_actual.append(below_prob)

                    # Generate buy/sell signals if below max positions
                    if len(open_positions) < max_positions:
                        if above_prob > decayed_threshold_prob:
                            open_positions.append((actual_prices[t], 1, above_prob, t))  # Include entry time `t`
                        elif below_prob > decayed_threshold_prob:
                            open_positions.append((actual_prices[t], -1, below_prob, t))  # Include entry time `t`

                    # Exit criteria check
                    closed_positions = []
                    for position in open_positions:
                        entry_price, position_type, entry_prob, entry_time = position
                        exit_price = actual_prices[t]
                        trade_return = (exit_price - entry_price) / entry_price if position_type == 1 else (entry_price - exit_price) / entry_price

                        if position_type == 1:
                            if trade_return >= profit_target or trade_return <= stop_loss or (entry_prob - above_prob)  >= decayed_decline_threshold:
                                total_trades += 1
                                accumulated_return += trade_return
                                total_return += ((actual_prices[t] - entry_price)  if position_type == 1
                                        else (entry_price - actual_prices[t]) )
                                average_total_principal.append(entry_price)
                                durations_list.append(t - entry_time)  # Track the duration of the trade
                                if trade_return > 0:
                                    successful_trades += 1
                                closed_positions.append(position)
                        elif position_type == -1:
                            if trade_return >= profit_target or trade_return <= stop_loss or (entry_prob - below_prob)  >= decayed_decline_threshold:
                                total_trades += 1
                                accumulated_return += trade_return
                                total_return += ((actual_prices[t] - entry_price)  if position_type == 1
                                        else (entry_price - actual_prices[t]) )
                                average_total_principal.append(entry_price)
                                durations_list.append(t - entry_time)  # Track the duration of the trade
                                if trade_return > 0:
                                    successful_trades += 1
                                closed_positions.append(position)

                    open_positions = [p for p in open_positions if p not in closed_positions]

                final_price = actual_prices[-1]
                for position in open_positions:
                    entry_price, position_type, _, entry_time = position
                    trade_return = (final_price - entry_price) / entry_price if position_type == 1 else (entry_price - final_price) / entry_price
                    total_trades += 1
                    accumulated_return += trade_return
                    total_return += ((actual_prices[t] - entry_price)  if position_type == 1
                                        else (entry_price - actual_prices[t]) )
                    average_total_principal.append(entry_price)
                    durations_list.append(forecast_horizon - entry_time)  # Track duration for unclosed positions
                    if trade_return > 0:
                        successful_trades += 1

            # Calculate performance for this run
            accuracy = successful_trades / total_trades if total_trades > 0 else 0
            average_duration = np.mean(durations_list) if durations_list else 0  # Calculate average duration if any trade occurred
            accuracy_list.append(accuracy)
            accumulated_return_list.append(accumulated_return)
            average_duration_list.append(average_duration)
            average_total_trades.append(total_trades)
            average_total_return.append(total_return)
    

        # Calculate and print average metrics across runs
        average_accuracy = np.mean(accuracy_list)
        average_accumulated_return = np.mean(accumulated_return_list)
        average_trade_duration = np.mean(average_duration_list)
        average_total_trades = np.mean(average_total_trades)
        average_total_return = np.mean(average_total_return)
        average_total_principal = np.mean(average_total_principal)

        return average_accuracy, np.mean(accumulated_return_list), np.mean(average_total_trades),average_trade_duration,average_total_return,average_total_principal,(average_total_return)/(average_total_principal*max_positions*0.3)

    # Run selected strategy with centered Lottie animation and spinner
    if st.button("Run Simulation"):
        # Create placeholders for the animation and results
        animation_placeholder = st.empty()
        results_placeholder = st.empty()

        # Display the Lottie animation in the center
        with animation_placeholder:
            st_lottie(
                lottie_animation,
                height=450,  # Increase or decrease to adjust size
                width=720,
                key="loading_animation",
            )

        # Add a funky spinner with rotating messages
        spinner_placeholder = st.empty()
        messages = itertools.cycle([
            "Analyzing market trends 📈",
            "Simulating trades 🛠️",
            "Crunching numbers 🧮",
            "Forecasting future prices 🔮",
            "Almost there... 🚀"
        ])
        
        # Simulate model execution with a spinner
        with st.spinner("Running the model. Please wait..."):
            for _ in range(10):  # Simulate the model's progress
                spinner_placeholder.text(next(messages))
                time.sleep(5)  # Adjust duration to fit the actual model runtime

            # Run the selected strategy
            if strategy == "Strategy 1":
                accuracy, accumulated_return, total_trades, avg_duration, tot_ret, avg_principal, avg_return = strategy_1(future_df, filtered_indices)
            elif strategy == "Strategy 2":
                accuracy, accumulated_return, total_trades, avg_duration, tot_ret, avg_principal, avg_return = strategy_2(future_df, filtered_indices)

        # Clear the animation and spinner as soon as results are ready
        animation_placeholder.empty()
        spinner_placeholder.empty()

        # Display the results in the results placeholder
        with results_placeholder.container():
            st.success("Simulation completed!")
            st.write(f"Average Accuracy: {accuracy:.2%}")
            st.write(f"Average Accumulated Return: {accumulated_return:.2%}")
            st.write(f"Average Total Trades: {total_trades:.2f}")
            st.write(f"Average Trade Duration over runs: {avg_duration:.2f} time steps")
            st.write(f"Total Return: {tot_ret}")
            st.write(f"Average Principal: {avg_principal}")
            st.write(f"Average Total Return %: {avg_return * 100:.2f}")
