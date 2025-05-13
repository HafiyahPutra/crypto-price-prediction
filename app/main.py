# main.py
from fastapi import FastAPI, HTTPException
import uvicorn
from app.data import get_ohlc_data
from app.model import prepare_data, build_lstm_model
from app.utils import plot_prediction
from app.config import COINS, WINDOW_SIZE, PRED_DAYS
import numpy as np
import os

# Import tambahan untuk load model dengan custom_objects dan load scaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse  # Import fungsi mse untuk custom_objects
import joblib  # Untuk load scaler, sesuaikan dengan cara Anda menyimpan scaler

app = FastAPI()

models = {}
scalers = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_and_save_model(coin_id):
    df = get_ohlc_data(coin_id)
    X, y, scaler = prepare_data(df)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    model_path = os.path.join(MODEL_DIR, f"{coin_id}_model.h5")
    model.save(model_path)
    # Simpan scaler secara terpisah menggunakan joblib
    scaler_path = os.path.join(MODEL_DIR, f"{coin_id}_scaler.save")
    joblib.dump(scaler, scaler_path)
    return model, scaler

def load_model_and_scaler(coin_id):
    model_path = os.path.join(MODEL_DIR, f"{coin_id}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{coin_id}_scaler.save")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # Load model dengan custom_objects untuk mse agar tidak error saat load
        model = load_model(model_path, custom_objects={'mse': mse})
        # Load scaler menggunakan joblib
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        # Jika model atau scaler belum ada, lakukan training dan simpan
        return train_and_save_model(coin_id)

@app.on_event("startup")
def startup_event():
    for coin in COINS.values():
        model, scaler = load_model_and_scaler(coin)
        models[coin] = model
        scalers[coin] = scaler

@app.get("/predict/{coin_name}")
def predict(coin_name: str):
    coin_id = COINS.get(coin_name.lower())
    if not coin_id:
        raise HTTPException(status_code=404, detail="Coin not supported")

    df = get_ohlc_data(coin_id)
    if len(df) < WINDOW_SIZE + PRED_DAYS:
        raise HTTPException(status_code=400, detail="Not enough data to make prediction")

    X, y, scaler = prepare_data(df)
    model = models.get(coin_id)
    if model is None:
        model, scaler = train_and_save_model(coin_id)
        models[coin_id] = model
        scalers[coin_id] = scaler

    last_input = X[-1].reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(last_input)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

    historical_close = df['close'].values

    img_base64 = plot_prediction(historical_close, pred, coin_name)

    return {
        "predicted_close_price": pred.tolist(),
        "plot_base64": img_base64
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
