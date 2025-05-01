
# Yahoo Finance API
# ==============================================================================
import yfinance as yf

# Handling and processing of Data
# ==============================================================================
import numpy as np
import pandas as pd
import scipy.stats as stats

# Handling and processing of Data for Date (time)
# ==============================================================================
import datetime
import time
from datetime import datetime, timedelta

# Statistics and plot
# ==============================================================================
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import normal_ad

# Plot
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from utilsforecast.plotting import plot_series
# Utils
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")



### Functions

def augmented_dickey_fuller_test(series):
    # Perform the Augmented Dickey-Fuller test
    dftest = adfuller(series, autolag='AIC')

    # Create a Series for the main test results
    dfoutput = pd.Series(
        dftest[0:4],
        index=['Test Statistic','p-value','No Lags Used','Number of Observations Used']
    )

    # Add critical values to the Series
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value

    # Convert the Series to a DataFrame for better display
    dfoutput_df = dfoutput.reset_index()
    dfoutput_df.columns = ['Parameter', 'Value']

    # Display the DataFrame as a table without the default index
    st.table(dfoutput_df)

    # Conclusion based on p-value
    if dftest[1] <= 0.05:
        st.subheader("Conclusion:")
        st.success("Reject the null hypothesis – The data is stationary.")
    else:
        st.subheader("Conclusion:")
        st.warning("Fail to reject the null hypothesis – The data is not stationary.")



# Function to plot autocorrelation and partial autocorrelation
def plot_autocorrelation(data):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Autocorrelation plot
    plot_acf(data["y"], lags=21, ax=axs[0], color="fuchsia")
    axs[0].set_title("Autocorrelation")

    # Partial Autocorrelation plot
    plot_pacf(data["y"], lags=21, ax=axs[1], color="lime")
    axs[1].set_title("Partial Autocorrelation")

    plt.tight_layout()
    st.pyplot(fig)

# Function to plot boxplot
def plot_boxplot(data, var):
    # Extract month names for grouping
    data_reset = data.reset_index()
    data_reset['Month'] = data_reset['ds'].dt.strftime('%B')  # Full month name
    # To ensure months are in calendar order
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.boxplot(x='Month', y=var, data=data_reset, order=months_order, palette='Set3', ax=ax)
    ax.set_title("Stock price Distribution per Month - Box Plot", fontsize=12)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel(var.capitalize(), fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)


def plot_lineplot(data):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    ax.set_title("Stock Price Lineplot", fontsize=12)

    ax.plot(data['ds'], data['y'], label='Stock Price', color='tab:blue')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    return fig


def main():
    st.title("Stock Analysis Dashboard")

    
    # Instructions with a clickable link
    st.markdown("""
    **Enter the Stock Ticker Symbol**

    For example, AMAZON is `AMZN`. If you need to look up a ticker symbol, visit [Yahoo Finance](https://finance.yahoo.com/).
    """)

    # Text input for ticker symbol
    selected_stock = st.text_input('Ticker Symbol:')

    # If no ticker symbol has been entered yet, display instructions
    if not selected_stock:
        st.info("Please enter a stock ticker symbol above to view data.")
        return  # Exit the function early if no ticker is specified

    # Display the entered ticker symbol
    if selected_stock:
        st.write(f"You entered: **{selected_stock.upper()}**")

    
    # Time selection
    time_options = ["5y","10y","15y","20y","25y","30y"]
    
    selected_time = st.selectbox("Select a horizon:", options=time_options)

    
    # Fetch data
    ticker = yf.Ticker(selected_stock)
    history = ticker.history(period=selected_time)

    if history.empty:
        st.error(f"No data found for ticker '{ticker_symbol}'. Please check the ticker symbol.")
        return

    data = history.copy()

    # Reset the index
    data.reset_index(inplace=True)

    # Ensure Date is a datetime object and format it
    data['Date'] = pd.to_datetime(data['Date'])

    # Select the necessary columns
    if 'Close' in data.columns:
        data = data[['Date', 'Close']]
    else:
        st.error("The fetched data does not contain a 'Close' column.")
        return

    # Rename the 'Date' column to 'ds' and 'Close' to 'y'
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    data["unique_id"] = "1"
    data.columns = ["ds", "y", "unique_id"]

    # Display header based on selected stock
    st.header(f"""{selected_stock} stock price - {selected_time}""") 
    st.write(data)

    st.header("Data Visualization")
    
    plot_lineplot(data)

    plot_boxplot(data, var='y')

    plot_autocorrelation(data)

    st.header("Time Series Decomposition")
    model = st.selectbox("Choose decomposition model:", ["additive", "multiplicative"])
    period = st.number_input("Enter period:", min_value=1, value=21)

    if st.button("Decompose"):
        try:
            result = seasonal_decompose(data["y"], model=model, period=period)
            fig = result.plot()
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.header("Dickey-Fuller test")
    augmented_dickey_fuller_test(data["y"])

if __name__ == "__main__":
    main()  

