from app.dss import tool_1, tool_2, tool_3
import pandas as pd

def test_fetch_data():
    # this function will return a list of datapoints
    data = pd.read_csv(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&datatype=csv')
    data = pd.read_csv()
    assert isinstance(data, list)
