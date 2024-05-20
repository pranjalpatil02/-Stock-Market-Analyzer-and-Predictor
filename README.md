# Stock Market Analyzer and Predictor

This is a Streamlit application that allows users to analyze and predict stock prices using machine learning models. The application fetches historical stock data, calculates moving averages, visualizes price trends, and predicts future stock prices based on the trained model.

## Features

- User authentication with Streamlit Login Auth UI
- Fetch historical stock data from Yahoo Finance API
- Calculate and visualize moving averages (MA50, MA100, MA200)
- Compare original stock prices with predicted prices
- Predict future stock prices for the next 15 days
- Fetch and display stock market news articles

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/stock-market-analyzer.git
```

2. Install the required dependencies:
 ```bash
   pip install streamlit streamlit-login-auth-ui yfinance keras matplotlib requests plotly scikit-learn streamlit-option-menu
```

3. Download the pre-trained machine learning model and place it in the project directory.

Usage

1. Run the Streamlit application:
```bash
   streamlit run app.py
```
2. Enter your login credentials to access the application.
3. Input the stock symbol you want to analyze.
4. Select the desired analysis or prediction option from the sidebar menu.
5. Explore the visualizations and predictions for the selected stock.

Configuration
Update the YOUR_API_KEY_HERE placeholder in the fetch_stock_news function with your NewsAPI API key to fetch stock market news articles.
Ensure that the pre-trained machine learning model file path in the main function is correct.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgments

Streamlit for the application framework
Yahoo Finance API for historical stock data
NewsAPI for stock market news articles
Keras for the machine learning library

This README provides an overview of the project, its features, installation instructions, usage guidelines, and information about contributing and licensing. You can customize it further based on your specific requirements or additional details you want to include.
