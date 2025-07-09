import pickle
import streamlit as st
import numpy as np
import pandas as pd
import time

# --- KONFIGURASI HALAMAN & GAYA (CSS) ---
st.set_page_config(
    page_title="Kalkulator Kalori",
    page_icon="",
    layout="centered"  # Menggunakan layout terpusat
)

# CSS Kustom untuk tema yang simpel dan modern
st.markdown("""
<style>
/* Mengubah font utama */
html, body, [class*="st-"] {
    font-family: 'Helvetica', 'Arial', sans-serif;
}

/* Warna latar belakang utama */
.stApp {
    background-color: #212121;
}

/* Gaya untuk judul utama */
h1 {
    color: #000000;
    font-weight: bold;
}

/* Gaya untuk kartu (card) */
.card {
    background-color: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Gaya untuk tombol */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    width: 100%;
}
.stButton>button:hover {
    background-color: #E03C3C;
}

/* Gaya untuk hasil metrik */
.result-metric {
    background-color: #28a745;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.result-metric h3 {
    color: white;
    margin-bottom: 5px;
}
.result-metric h1 {
    color: white;
    font-size: 3.5rem;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# --- LOGIKA APLIKASI ---

# Inisialisasi variabel
model = None
scaler = None

# Nama file model dan scaler
NAMA_FILE_MODEL = 'model_kalori.pkl'
NAMA_FILE_SCALER = 'scaler_kalori.sav'

# Muat scaler dan model menggunakan cache agar lebih cepat
@st.cache_resource
def load_model_scaler():
    try:
        with open(NAMA_FILE_SCALER, 'rb') as file_scaler:
            scaler = pickle.load(file_scaler)
        with open(NAMA_FILE_MODEL, 'rb') as file_model:
            model = pickle.load(file_model)
        return model, scaler
    except FileNotFoundError:
        st.error(f"GAGAL MEMUAT FILE: Pastikan file '{NAMA_FILE_MODEL}' dan '{NAMA_FILE_SCALER}' ada.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi error: {e}")
        return None, None

model, scaler = load_model_scaler()


# --- TAMPILAN APLIKASI (UI) ---

st.title("🎯 Kalkulator Estimasi Kalori")
st.write("Analisis kandungan nutrisi makro makanan Anda untuk mendapatkan estimasi total kalorinya.")
st.markdown("---")


# Gunakan st.form untuk input dalam sebuah "kartu"
st.markdown('<div class="card" color="yellow">', unsafe_allow_html=True)
with st.form(key="calorie_form"):
    st.subheader("👇 Masukkan Detail Nutrisi")
    
    col1, col2 = st.columns(2)
    with col1:
        fat = st.number_input("Lemak (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
        protein = st.number_input("Protein (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    with col2:
        carb = st.number_input("Karbohidrat (g)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
        sugars = st.number_input("Gula (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    
    st.write("")
    submit_button = st.form_submit_button(label='✨ Hitung Sekarang!')
st.markdown('</div>', unsafe_allow_html=True)


# --- PROSES DAN TAMPILAN HASIL ---
if submit_button:
    if model is not None and scaler is not None:
        with st.spinner('Model sedang menganalisis...'):
            time.sleep(1) # Memberi jeda agar spinner terlihat
            # 1. Siapkan data input mentah
            user_input_raw = pd.DataFrame({
                'calories': [0], 'fat': [fat], 'carb': [carb], 'sugars': [sugars], 'protein': [protein]
            })
            user_input_raw = user_input_raw[['calories', 'fat', 'carb', 'sugars', 'protein']]

            # 2. Lakukan penskalaan (transform)
            input_scaled = scaler.transform(user_input_raw)

            # 3. Ambil fitur untuk prediksi
            features_for_prediction = [input_scaled[0, 1], input_scaled[0, 2], input_scaled[0, 4], input_scaled[0, 3]]
            
            # 4. Lakukan prediksi
            prediction_scaled = model.predict([features_for_prediction])
            
            # 5. Lakukan penskalaan balik (inverse transform)
            dummy_for_inverse = np.zeros((1, 5))
            dummy_for_inverse[0, 0] = prediction_scaled[0]
            prediction_unscaled = scaler.inverse_transform(dummy_for_inverse)
            kalori_sebenarnya = prediction_unscaled[0, 0]

        # Tampilkan hasil dalam "kartu" terpisah
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Hasil Estimasi Anda")
        
        # Tampilkan hasil akhir dengan gaya khusus
        st.markdown(f"""
        <div class="result-metric">
            <h3>Total Estimasi Kalori</h3>
            <h1>{kalori_sebenarnya:.2f} kkal</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Gunakan expander untuk detail input
        with st.expander("Lihat detail nutrisi yang Anda masukkan"):
            st.write(f"- **Lemak:** `{fat}` g")
            st.write(f"- **Karbohidrat:** `{carb}` g")
            st.write(f"- **Protein:** `{protein}` g")
            st.write(f"- **Gula:** `{sugars}` g")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("Gagal melakukan prediksi. Pastikan file model dan scaler sudah dimuat dengan benar.")