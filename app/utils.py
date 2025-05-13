# utils.py
import matplotlib.pyplot as plt
import io
import base64

def plot_prediction(historical_prices, predicted_prices, coin_name):
    plt.figure(figsize=(10,5))
    plt.plot(range(len(historical_prices)), historical_prices, label='Historical Close Price')
    plt.plot(range(len(historical_prices), len(historical_prices) + len(predicted_prices)), predicted_prices, label='Predicted Close Price')
    plt.legend()
    plt.title(f'Close Price Prediction for {coin_name.upper()}')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64
