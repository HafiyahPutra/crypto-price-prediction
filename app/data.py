# data.py
import requests
import pandas as pd
from app.config import API_KEY, BASE_URL, VS_CURRENCY, DAYS

def get_ohlc_data(coin_id):
    url = f"{BASE_URL}/{coin_id}/ohlc?vs_currency={VS_CURRENCY}&days={DAYS}"
    headers = {
        "x-cg-demo-api-key": API_KEY
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Data format: [timestamp, open, high, low, close]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
