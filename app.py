from streamlit_login_auth_ui.widgets import __login__
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import requests
import json
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

model = load_model('D:\STOCKS PRED\Stock Predictions Model(19).keras')
model.summary()

__login__obj = __login__(
    auth_token="courier_auth_token",
    company_name="Shims",
    width=200,
    height=250,
    logout_button_name='Logout',
    hide_menu_bool=False,
    hide_footer_bool=False,
    lottie_url='https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json'
)

LOGGED_IN = __login__obj.build_login_ui()

# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Check if the user is logged in
if LOGGED_IN == True:
    # Set the logged_in attribute to True if the user is logged in
    st.session_state.logged_in = True
    # Start your Streamlit application here
    st.markdown("WELCOME!")

@st.cache_data
def main():
    model = load_model('D:\stock market pred\Stock Predictions Model(1).keras')
    model.summary()

# Function to fetch daily stock data
def fetch_daily_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to plot Price vs MA50 chart
def plot_price_vs_ma50(data, stock_name, stock_symbol):
    ma_50_days = data.Close.rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Original Price', line=dict(color='green')))
    fig.update_layout(title=f'{stock_name} ({stock_symbol}) Price vs MA50', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig

# Function to plot Price vs MA50 vs MA100 chart
def plot_price_vs_ma50_ma100(data, stock_name, stock_symbol):
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Original Price', line=dict(color='green')))
    fig.update_layout(title=f'{stock_name} ({stock_symbol}) Price vs MA50 vs MA100', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig

# Function to plot Price vs MA100 vs MA200 chart
def plot_price_vs_ma100_ma200(data, stock_name, stock_symbol):
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Original Price', line=dict(color='green')))
    fig.update_layout(title=f'{stock_name} ({stock_symbol}) Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig

# Function to predict future stock prices
def predict_future_prices(data, scaler, model):
    future_predictions = []
    last_100_days = data['Close'][-100:].values.reshape(1, -1)

    for _ in range(15):
        # Scale the last 100 days of data
        last_100_days_scaled = scaler.transform(last_100_days.reshape(-1, 1))
        # Reshape the scaled data to match the model input shape
        last_100_days_scaled = last_100_days_scaled.reshape(1, 100, 1)
        # Predict the next day's price
        future_prediction = model.predict(last_100_days_scaled)
        # Denormalize the predicted price
        future_prediction = future_prediction * scaler.scale_ + scaler.min_
        # Append the predicted price to the list
        future_predictions.append(future_prediction[0, 0])
        # Update the last 100 days of data for the next prediction
        last_100_days = np.append(last_100_days[:, 1:], future_prediction[0]).reshape(1, -1)

    # Generate future dates for the next 15 days starting from today
    today_date = datetime.now().strftime('%Y-%m-%d')
    future_dates = [(datetime.strptime(today_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 16)]

    # Create a DataFrame to hold future predictions and dates
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

    # Plot the predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines+markers', name='Predicted Price', line=dict(color='blue')))
    fig.update_layout(title='Predicted Future Prices', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    
    # Additional information about future predictions
    st.write("Here are the predicted future prices for the next 15 days:")
    st.write(future_df)
    st.write("These predictions are based on historical data and a machine learning model trained on past stock performance.")

    # Show the chart
    st.plotly_chart(fig)

# Function to plot Original Price vs Predicted Price chart
def plot_original_vs_predicted(data_test_scale, scaler, model, stock_name, stock_symbol):
    # Define x and y for prediction
    x = []
    y = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])
    x, y = np.array(x), np.array(y)

    # Predict using the model
    predict = model.predict(x)
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    # Create a DataFrame for original and predicted prices
    df = pd.DataFrame({'Original Price': y, 'Predicted Price': predict.flatten()})

    # Plot Original Price vs Predicted Price as a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(predict.flatten())), y=predict.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
    fig.update_layout(title=f'{stock_name} ({stock_symbol}) Original Price vs Predicted Price', xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
    
    # Return the Plotly chart object
    return fig

# Function to fetch stock news
def fetch_stock_news(stock_symbol):
    # Define the API URL for fetching stock news
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey=YOUR_API_KEY_HERE"
    
    # Make the HTTP GET request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Try to parse the JSON response
            news_data = response.json()
            articles = news_data['articles']
            return articles
        except json.JSONDecodeError as e:
            # Handle the case when the response is not in valid JSON format
            print(f"Failed to parse JSON response: {e}")
            return []
    else:
        # Handle the case when the request fails
        print(f"Failed to fetch news articles. Status code: {response.status_code}")
        return []

# Define the main function
def main():
    st.title('STOCK MARKET ANALYSER & PREDICTION')

    
    if st.session_state.logged_in:
        # Input stock symbol
        stock_symbol = st.text_input('Enter Stock Symbol', 'GOOG')

        # Fetch stock name
        stock_name = yf.Ticker(stock_symbol).info['longName']

        # Define start and end dates for daily data (adjust as needed)
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch daily stock data
        data = fetch_daily_data(stock_symbol, start_date, end_date)

        # Continue with the rest of your code...
        # Define data_train and data_test
        data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

        # Define scaler and scale data_test
        scaler = MinMaxScaler(feature_range=(0,1))
        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        st.title("Main Page")
        st.sidebar.success("Select A Page Above")

        st.sidebar.title('Navigation')
        page = st.sidebar.radio("Go to", ['Historical Stock Data', 'Price vs MA50', 'Price vs MA50 vs MA100', 'Price vs MA100 vs MA200', 'Original Price vs Predicted Price', 'Predicted Future Prices', 'Stock Market News'], key='main_radio')
        
        if page == 'Historical Stock Data':
            st.subheader('Historical Stock Data')
            st.write(data)
            # Main functionality goes here (e.g., predictions, charts, etc.
        elif page == 'Price vs MA50':
            st.subheader('Price vs MA50')
            fig = plot_price_vs_ma50(data, stock_name, stock_symbol)
            st.plotly_chart(fig)
        elif page == 'Price vs MA50 vs MA100':
            st.subheader('Price vs MA50 vs MA100')
            fig = plot_price_vs_ma50_ma100(data, stock_name, stock_symbol)
            st.plotly_chart(fig)
        elif page == 'Price vs MA100 vs MA200':
            st.subheader('Price vs MA100 vs MA200')
            fig = plot_price_vs_ma100_ma200(data, stock_name, stock_symbol)
            st.plotly_chart(fig)
        elif page == 'Original Price vs Predicted Price':
            st.subheader('Original Price vs Predicted Price')
            fig = plot_original_vs_predicted(data_test_scale, scaler, model, stock_name, stock_symbol)
            st.plotly_chart(fig)
        elif page == 'Predicted Future Prices':
            st.subheader('Predicted Future Prices')
            predict_future_prices(data, scaler, model)
        elif page == 'Stock Market News':
            st.subheader('Stock Market News')
            news_articles = fetch_stock_news(stock_symbol)
            if news_articles:
                for article in news_articles:
                    st.write(f"**{article['title']}**")
                    st.write(article['description'])
                    st.write(f"Source: [{article['source']['name']}]({article['url']})")
                    st.write("---")
            else:
                st.write("No news articles found for this stock symbol.")

if __name__ == '__main__':
    main()
