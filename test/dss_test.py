import pandas as pd
import os

def test_fetch_data():
    # this function will return a list of datapoints
    API_KEY = os.getenv("API_KEY")
    symbol="C"
    data = pd.read_csv(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&datatype=csv')
    data = pd.read_csv()
    assert isinstance(data, list)
