# config.py
API_KEY = "CG-7pi9DCcf6E6PmCFBLrwvGtZT"
BASE_URL = "https://api.coingecko.com/api/v3/coins"
VS_CURRENCY = "usd"
DAYS = 60      # Jumlah hari data historis yang diambil

WINDOW_SIZE = 7       # Jumlah hari data input untuk prediksi
PRED_DAYS = 7         # Jumlah hari yang diprediksi ke depan

COINS = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "binancecoin": "binancecoin",
    "ripple": "ripple",
    "solana": "solana"
}
