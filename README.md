Tentu! Berikut adalah contoh file README.md yang lengkap dan terstruktur untuk proyek AI prediksi harga cryptocurrency Anda menggunakan LSTM, dengan data OHLC dari CoinGecko API, serta deployment API menggunakan FastAPI dan Render.

---

# README.md

```markdown
# Crypto Price Prediction API

Proyek ini adalah aplikasi API untuk memprediksi harga cryptocurrency (BTC, ETH, BNB, XRP, SOLANA) 7 hari ke depan menggunakan model LSTM berbasis data OHLC (Open, High, Low, Close) selama 30 hari terakhir. Data diambil secara real-time dari CoinGecko API.

API ini dibangun menggunakan Python, FastAPI, dan TensorFlow, dan dapat dideploy dengan mudah di platform Render.

---

## Fitur Utama

- Prediksi harga **Close Price** cryptocurrency 7 hari ke depan.
- Model LSTM yang dilatih menggunakan data OHLC 30 hari terakhir.
- Mendukung koin: BTC, ETH, BNB, XRP, SOLANA.
- API RESTful dengan FastAPI untuk integrasi mudah ke aplikasi web.
- Deployment mudah di Render.

---
## Cara Instalasi dan Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/crypto-price-prediction.git
cd crypto-price-prediction
```

### 2. Buat Virtual Environment dan Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Jalankan Training Model (Opsional)

Jika Anda ingin melatih ulang model:

```bash
python train.py
```

> Catatan: Training akan menggunakan data OHLC 30 hari terakhir dari CoinGecko API.

### 4. Jalankan API Server

```bash
uvicorn app.main:app --reload
```

API akan berjalan di `http://127.0.0.1:8000`.

---

## Endpoint API

- **GET /**  
  Menampilkan halaman dokumentasi Swagger UI.

- **POST /predict**  
  Menerima input nama koin (misal: "bitcoin") dan mengembalikan prediksi harga Close Price 7 hari ke depan.

### Contoh Request

```json
{
  "coin": "bitcoin"
}
```

### Contoh Response

```json
{
  "coin": "bitcoin",
  "predictions": [45000.12, 45230.45, 45500.78, 45780.34, 46000.00, 46250.12, 46500.45]
}
```

---

## Penjelasan Metode

Model menggunakan **Long Short-Term Memory (LSTM)**, jenis Recurrent Neural Network (RNN) yang sangat efektif untuk data deret waktu seperti harga cryptocurrency.

### Kelebihan LSTM

- Mampu menangkap pola jangka panjang dalam data deret waktu.
- Mengatasi masalah vanishing gradient pada RNN standar.
- Cocok untuk prediksi harga yang bergantung pada data historis.

### Kekurangan LSTM

- Membutuhkan data dan waktu pelatihan yang cukup banyak.
- Model bisa overfitting jika tidak diatur dengan baik.
- Memerlukan tuning hyperparameter yang cermat.

---

## Deployment di Render

1. Push kode ke GitHub.
2. Buat aplikasi baru di Render dengan koneksi ke repository GitHub.
3. Atur build command:

```bash
pip install -r requirements.txt
```

4. Atur start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

5. Deploy dan tunggu proses selesai.

---

## Kontribusi

Jika Anda ingin berkontribusi, silakan buat pull request atau buka issue.

---

## Lisensi

Proyek ini menggunakan lisensi MIT.

---

## Kontak

Untuk pertanyaan atau bantuan, silakan hubungi:

- Email: haffiyanputra@gmail.com
- GitHub: [https://github.com/HafiyahPutra](https://github.com/HafiyahPutra)


```

---

Jika Anda ingin saya buatkan versi README yang lebih singkat, lebih teknis, atau dengan tambahan bagian lain, silakan beritahu saya! 😊
