# Yahoo Finance API
# ==============================================================================
import yfinance as yf
# Handling and processing of Data
# ==============================================================================
import numpy as np
import pandas as pd

# Handling and processing of Data for Date (time)
# ==============================================================================
import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller

# Statistics and plot
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Neuralforecast
#===============================================================================
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import PredictionIntervals
from neuralforecast.losses.pytorch import DistributionLoss, MAE
import torch
from sklearn.metrics import mean_squared_error

# Utils
# ==============================================================================
import re
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# Instructions with a clickable link
st.markdown("""
**Enter the Stock Ticker Symbol**

For example, AMAZON is `AMZN`. If you need to look up a ticker symbol, visit [Yahoo Finance](https://finance.yahoo.com/).
""")

# Text input for ticker symbol
selected_stock = st.text_input('Ticker Symbol:')

# Time selection
time_options = ["5y", "10y", "15y", "20y", "25y", "30y"]
selected_time = st.selectbox("Select a horizon:", options=time_options)

# Coverage selection using number input
selected_coverage = st.number_input(
            "Enter Coverage Level (%):",
            min_value=50.0,
            max_value=99.0,
            value=95.0,
            step=0.5,
            format="%.1f"
)

# Fetch data
if selected_stock:
    stock_ticker = yf.Ticker(selected_stock)
    stock_history = stock_ticker.history(period=selected_time)

    # Check if the history DataFrame is empty
    if stock_history.empty:
        st.error(f"No data found for ticker '{selected_stock.upper()}'. Please check the ticker symbol or the selected time range.")
    else:
        # Display the entered ticker symbol
        st.write(f"You entered: **{selected_stock.upper()}**")
        
        # Display time selection
        st.write(f"Selected time period:**{selected_time.upper()}**")


        # Ensure coverage is within bounds
        if selected_coverage < 50.0 or selected_coverage > 99.0:
            st.error("Please enter a coverage level between 50% and 99%.")

        # Display selected coverage
        st.write(f"Selected Coverage Level:**{selected_coverage:.1f}**%")


        df = stock_history.copy()
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df["unique_id"] = "1"
        df.columns = ["ds", "y", "unique_id"]

        # Continue with data preparation and modeling...
        test_size = 63
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]

        # Display split data
        st.markdown("""**Split data**""")
        st.write(f"Train Start Date: {train['ds'].min().date()}")
        st.write(f"Train End Date: {train['ds'].max().date()}")
        st.write(f"Test Start Date: {test['ds'].min().date()}")
        st.write(f"Test End Date: {test['ds'].max().date()}")
        st.write(f"Projection: 1 month")

        # Header dynamically based on selection
        st.header(f"{selected_stock} Stock Price - {selected_time} - {selected_coverage}% PIs with NHITS")


        def rename_forecast_columns(df):
            new_columns = {}
            for col in df.columns:
                if col == 'ds':
                    continue
                elif col == 'unique_id':
                    continue
                elif col == 'y':
                    continue
                elif re.search(r'-lo-\d+', col):
                    new_columns[col] = 'y_hat_lower'
                elif re.search(r'-hi-\d+', col):
                    new_columns[col] = 'y_hat_upper'
                else:
                    new_columns[col] = 'y_hat'
            df = df.rename(columns=new_columns)
            return df

        def calculate_rmse(predictions, targets):
            return np.sqrt(mean_squared_error(targets, predictions))

        def evaluate_metrics_with_boundaries(df):
            rmse = calculate_rmse(df['y_hat'], df['y'])
            rmse_lo = calculate_rmse(df['y_hat_lower'], df['y'])
            rmse_hi = calculate_rmse(df['y_hat_upper'], df['y'])
            coverage = ((df['y'] >= df['y_hat_lower']) & (df['y'] <= df['y_hat_upper'])).mean()
            average_width = (df['y_hat_upper'] - df['y_hat_lower']).mean()
            metrics = {'RMSE_Point': rmse, 'RMSE_Lower_Bound': rmse_lo, 'RMSE_Upper_Bound': rmse_hi, 'Coverage (%)': coverage * 100, 'Average Width': average_width}
            return metrics

        def forecast_vis(train, test, prediction, ticker):
            train_df = train.reset_index()
            test_df = test.reset_index()
            prediction_df = prediction.reset_index()
            train_df['ds'] = pd.to_datetime(train_df['ds'])
            test_df['ds'] = pd.to_datetime(test_df['ds'])
            prediction_df['ds'] = pd.to_datetime(prediction_df['ds'])
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(train_df['ds'], train_df['y'], label='Train Actual', color='black', linewidth=2)
            ax.plot(test_df['ds'], test_df['y'], label='Test Actual', color='blue', linewidth=2)
            ax.plot(test_df['ds'], test_df['y_hat'], label='Test Prediction', color='orange', linewidth=2)
            ax.fill_between(test_df['ds'], test_df['y_hat_lower'], test_df['y_hat_upper'], color='orange', alpha=0.2, label='Test Prediction Interval')
            ax.plot(prediction_df['ds'], prediction_df['y_hat'], label='Future Prediction', color='green', linewidth=2)
            ax.fill_between(prediction_df['ds'], prediction_df['y_hat_lower'], prediction_df['y_hat_upper'], color='green', alpha=0.2, label='Future Prediction Interval')
            ax.set_xlabel('Date', fontsize=15)
            ax.set_ylabel('Value', fontsize=15)
            ax.set_title(f'Forecasting Results: {ticker}', fontsize=20)
            ax.legend(fontsize=15)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.tight_layout()
            st.pyplot(fig)

        def forecast_vis2(test, prediction, ticker):
            test_df = test_result.reset_index(drop=True)
            prediction_df = prediction_result.reset_index(drop=True)
            test_df['ds'] = pd.to_datetime(test_df['ds'])
            prediction_df['ds'] = pd.to_datetime(prediction_df['ds'])
            fig = plt.figure(figsize=(16, 8))
            plt.plot(test_df['ds'], test_df['y'], label='Test Actual', color='blue', linewidth=2)
            plt.plot(test_df['ds'], test_df['y_hat'], label='Test Prediction', color='orange', linewidth=2)
            plt.fill_between(test_df['ds'], test_df['y_hat_lower'], test_df['y_hat_upper'], color='orange', alpha=0.2, label='Test Prediction Interval')
            plt.plot(prediction_df['ds'], prediction_df['y_hat'], label='Future Prediction', color='green', linewidth=2)
            plt.fill_between(prediction_df['ds'], prediction_df['y_hat_lower'], prediction_df['y_hat_upper'], color='green', alpha=0.2, label='Future Prediction Interval')
            plt.xlabel('Date', fontsize=15)
            plt.ylabel('Value', fontsize=15)
            plt.title(f'Test and Future Conformal Prediction Results {ticker}', fontsize=20)
            plt.legend(fontsize=15)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            plt.tight_layout()
            st.pyplot(fig)

        # Initializing and Fitting the Model
        horizon = 84
        input_size = 3  # This sets n_windows

        # Initialize the NHITS model with the specified input_size
        models = [
            NHITS(
            h=horizon,
            input_size=input_size*horizon,
            max_steps=10,
            dropout_prob_theta=0.6,
            loss=DistributionLoss("Normal", level=[95]),
            scaler_type="robust",
            random_seed=1
            )
        ]

        # Initialize NeuralForecast with the model and transformations
        nf = NeuralForecast(
        models=models,
        freq='B'
        )

        # Fit the model on the training data with a validation size equal to the horizon
        nf.fit(train, val_size=horizon)

        # Generate forecasts
        forecast_df = nf.predict(train, level=[95])
        forecast_df=rename_forecast_columns(forecast_df)

        test_max_date = test['ds'].max()
        test_forecast = forecast_df[forecast_df['ds'] <= test_max_date]
        test_result = pd.merge(
            test_forecast,
            test,
            on=['unique_id', 'ds'],
            how='left'
        )
        test_result = test_result.dropna()
        prediction_result = forecast_df[forecast_df['ds'] > test_max_date]

        test_metrics = evaluate_metrics_with_boundaries(test_result)
        st.write("**Test metrics:**")
        for key, value in test_metrics.items():
            st.write(f"{key}: {value:.4f}")

        forecast_vis(train, test_result, prediction_result, selected_stock)
        forecast_vis2(test_result, prediction_result, selected_stock)