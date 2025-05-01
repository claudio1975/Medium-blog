import streamlit as st
import yfinance as yf
from autogen import ConversableAgent
from datetime import datetime
import pandas as pd

# ------------------------------
# 1. Page Configuration
# ------------------------------
st.set_page_config(page_title="Stock Agent Analysis", layout="centered")

# ------------------------------
# 2. User Instructions
# ------------------------------
st.markdown("""
### **Stock Agent Analysis**
**Enter the Stock Ticker Symbol and the OpenAI API_Key**

For example, AMAZON is `AMZN`. If you need to look up a ticker symbol, visit [Yahoo Finance](https://finance.yahoo.com/).
""")

# ------------------------------
# 3. Input: Ticker Symbol
# ------------------------------
selected_stock = st.text_input('Ticker Symbol:', value='')

# API Key input
api_key = st.text_input("Please Copy & Paste your API_KEY", key="chatbot_api_key", type="password")

if selected_stock:
    selected_stock = selected_stock.upper()  # Ensure uppercase
    st.write(f"You selected: **{selected_stock}**")

    # ------------------------------
    # 4. Input: Time Horizon Selection
    # ------------------------------
    time_options = ["1mo", "3mo", "6mo", "1y"]
    selected_time = st.selectbox("Select Time Horizon:", options=time_options, index=3)  # Default to "1y"


    # ------------------------------
    # 5. Fetch Historical Data
    # ------------------------------
    def fetch_history(ticker_symbol, period):
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    history = fetch_history(selected_stock, selected_time)

    if history is not None and not history.empty:
        # ------------------------------
        # 6. Display Latest Date
        # ------------------------------
        max_date = history.index.max().strftime('%Y-%m-%d')
        st.write(f"Research Date (Latest Data): **{max_date}**")

        # ------------------------------
        # 7. Display Historical Data
        # ------------------------------
        with st.expander("View Historical month Data"):
            st.dataframe(history.tail(30))  # Show last 30 entries

        # ------------------------------
        # 8. Process Historical Data for Agent
        # ------------------------------
        # Extract key metrics or a data summary to include in the prompt
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def process_history_data(hist):
            # Example: Calculate Moving Averages and RSI
            hist['20_SMA'] = hist['Close'].rolling(window=20).mean()
            hist['50_SMA'] = hist['Close'].rolling(window=50).mean()
            hist['100_SMA'] = hist['Close'].rolling(window=100).mean()
            hist['180_SMA'] = hist['Close'].rolling(window=180).mean()
            hist['250_SMA'] = hist['Close'].rolling(window=250).mean()
            hist['RSI'] = calculate_rsi(hist)

            # Extract latest indicators
            latest_data = hist.iloc[-1]
            indicators = {
                'Current Price': latest_data['Close'],
                '20-day SMA': latest_data['20_SMA'],
                '50-day SMA': latest_data['50_SMA'],
                '100-day SMA': latest_data['100_SMA'],
                '180-day SMA': latest_data['180_SMA'],
                '250-day SMA': latest_data['250_SMA'],
                'RSI': latest_data['RSI'],
                'Volume': latest_data['Volume']
            }

            # Convert indicators to a readable format
            indicators_text = "\n".join([f"- **{key}**: {value:.2f}" for key, value in indicators.items() if pd.notna(value)])
            return indicators_text

        indicators_summary = process_history_data(history)

        # ------------------------------
        # 9. LLM Configuration
        # ------------------------------
        llm_api_key = api_key
        if not llm_api_key:
            st.error("API key for OpenAI is missing. Please set it in Streamlit box.")
        else:
            llm_config = {
                "model": "gpt-4o-mini",  # Ensure the model name is correct
                "api_key": llm_api_key
            }

            # ------------------------------
            # 10. Initialize ConversableAgent
            # ------------------------------
            try:
                agent = ConversableAgent(
                    name="Chatbot",
                    llm_config=llm_config,  # The Agent will use the LLM config provided to answer
                    human_input_mode="NEVER",  
                )
            except Exception as e:
                st.error(f"Error initializing ConversableAgent: {e}")
                agent = None

            # ------------------------------
            # 11. Define the Recommendation Task
            # ------------------------------
            task = (
                f"Analyze the following historical stock price data for **{selected_stock}** up to **{max_date}**.\n\n"
                f"**Key Indicators:**\n{indicators_summary}\n\n"
                "Based on this data, provide a clear recommendation on whether the stock is **Bullish** or **Neutral** or **Bearish**. "
                "Include relevant indicators and a brief explanation to support your recommendation."
                "You don't make predictions, only reccomendations by analysis"
            )

            # ------------------------------
            # 12. Function to Generate Recommendation
            # ------------------------------
            def generate_recommendation(agent, task):
                try:
                    with st.spinner("Generating recommendation..."):
                        reply = agent.generate_reply(
                            messages=[
                                {
                                    "role": "user",
                                    "content": task
                                }
                            ]
                        )
                    return reply
                except Exception as e:
                    st.error(f"An error occurred while generating the recommendation: {e}")
                    return None

            # ------------------------------
            # 13. Button to Trigger Recommendation Generation
            # ------------------------------
            if st.button("Generate Recommendation"):
                if agent:
                    recommendation = generate_recommendation(agent, task)
                    if recommendation:
                        st.markdown("### **Recommendation:**")
                        st.write(recommendation)
                else:
                    st.error("ConversableAgent is not initialized properly.")
    else:
        st.error("No historical data found for the provided ticker symbol. Please check the symbol and try again.")