import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Delivery Time Predictor", layout="centered")

st.title("🚚 Delivery Time Prediction App")
st.write("Aplikasi ini memprediksi durasi waktu pengiriman makanan menggunakan model XGBoost berdasarkan 9 fitur optimal yang telah Anda latih.")

# Load model, scaler, dan encoder
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_delivery.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
    st.success("Artifact model, scaler, dan encoder berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model/scaler/encoder: {e}")

st.subheader("Masukkan Detail Pesanan")

# 1. Input Kategori Toko
store_categories = list(encoder.categories_[0]) if 'encoder' in locals() else ["american", "mexican", "fast", "sandwich"]
selected_category = st.selectbox("Kategori Utama Toko (store_primary_category)", options=store_categories)

col1, col2 = st.columns(2)

with col1:
    total_items = st.number_input("Total Items", min_value=1, max_value=500, value=3)
    subtotal = st.number_input("Subtotal Harga (sen)", min_value=0, max_value=100000, value=2660)
    num_distinct_items = st.number_input("Jumlah Jenis Item", min_value=1, max_value=100, value=2)
    min_item_price = st.number_input("Harga Item Termurah", min_value=0, max_value=50000, value=600)

with col2:
    max_item_price = st.number_input("Harga Item Termahal", min_value=0, max_value=50000, value=1100)
    total_onshift_partners = st.number_input("Total Mitra Onshift", min_value=0, max_value=500, value=44)
    total_busy_partners = st.number_input("Total Mitra Sibuk", min_value=0, max_value=500, value=41.0)
    total_outstanding_orders = st.number_input("Total Outstanding Orders", min_value=0, max_value=500, value=56.0)

# Tombol Prediksi
if st.button("Prediksi Waktu Pengiriman"):
    try:
        # 1. Encode kategori toko pilihan pengguna menggunakan OrdinalEncoder
        category_df = pd.DataFrame([[selected_category]], columns=['store_primary_category'])
        category_encoded = encoder.transform(category_df)[0][0]
        
        # 2. Gabungkan input mentah menjadi DataFrame
        input_raw = pd.DataFrame([{
            "category_encoded": category_encoded,
            "total_items": total_items,
            "subtotal": subtotal,
            "num_distinct_items": num_distinct_items,
            "min_item_price": min_item_price,
            "max_item_price": max_item_price,
            "total_onshift_partners": total_onshift_partners,
            "total_busy_partners": total_busy_partners,
            "total_outstanding_orders": total_outstanding_orders
        }])

        # 3. Lakukan scaling input menggunakan scaler bawaan
        input_scaled = pd.DataFrame(
            scaler.transform(input_raw),
            columns=input_raw.columns
        )

        # 4. Prediksi menggunakan XGBoost
        prediction_seconds = model.predict(input_scaled)[0]
        prediction_minutes = prediction_seconds / 60.0

        st.balloons()
        st.success(f"⏱️ Estimasi Waktu Pengiriman: **{prediction_minutes:.2f} menit** ({prediction_seconds:.0f} detik)")
    except Exception as err:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {err}")
