import pickle
import streamlit as st
import numpy as np
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="XG TRACKER",
    page_icon="",
    layout="centered"
)

# --- LOGIKA & DATA ---
@st.cache_resource
def load_dependencies():
    """Memuat model dan scaler."""
    try:
        with open('scaler_kalori.sav', 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        with open('model_kalori.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        return model, scaler
    except FileNotFoundError:
        st.error("Gagal memuat file model/scaler. Pastikan file ada.")
        return None, None

model, scaler = load_dependencies()

def calculate_food_calories(fat, carb, protein, sugars):
    """Memprediksi kalori dari makanan."""
    if not model or not scaler:
        return 0
    user_input = pd.DataFrame({
        'calories': [0], 'fat': [fat], 'carb': [carb], 'sugars': [sugars], 'protein': [protein]
    })
    input_scaled = scaler.transform(user_input)
    features = [input_scaled[0, 1], input_scaled[0, 2], input_scaled[0, 4], input_scaled[0, 3]]
    prediction = model.predict([features])
    dummy_inverse = np.zeros((1, 5))
    dummy_inverse[0, 0] = prediction[0]
    return scaler.inverse_transform(dummy_inverse)[0, 0]

# --- UI & CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #1A1C20;
    color: #EAEAEA;
}
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
}

.input-container {
    background-color: #272A30;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #3A3D46;
}

.progress-ring {
    position: relative;
    width: 220px;
    height: 220px;
    margin: 20px auto;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: conic-gradient(#8A42D1 var(--progress-angle), #3A3D46 var(--progress-angle));
    transition: background 0.5s ease;
}
.progress-ring::before {
    content: '';
    position: absolute;
    width: 85%;
    height: 85%;
    background-color: #272A30;
    border-radius: 50%;
}
.progress-text {
    position: relative;
    text-align: center;
}
.progress-text h2 {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    color: #8A42D1;
}
.progress-text p {
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
    color: #A0AEC0;
}

.stButton>button {
    background-color: #8A42D1;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px 24px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


## --- BAGIAN UTAMA APLIKASI ---

st.title("🎯 XG TRACKER")

# Inisialisasi state
if 'total_calories_consumed' not in st.session_state:
    st.session_state.total_calories_consumed = 0

# Dummy value untuk tampilan awal
daily_goal = 2000 

# Bagian Laporan (ditampilkan lebih dulu)
with st.container():
    st.subheader("Sisa Kalori Harian Anda")
    report_placeholder = st.empty()

## Semua input dibungkus dalam satu kontainer "kartu"
with st.container():

    # 1. Atur Target
    st.subheader("1. Atur Target Kalori Harian")
    daily_goal = st.number_input("Target Kalori (kkal)", min_value=0, max_value=5000, value=2000, step=50, label_visibility="collapsed")

    st.markdown("---")

    # 2. Input Data Makanan
    st.subheader("2. Hitung Kalori")
    
    col1, col2 = st.columns(2)
    with col1:
        f_fat = st.number_input("Lemak (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
        f_protein = st.number_input("Protein (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    with col2:
        f_carb = st.number_input("Karbohidrat (g)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
        f_sugars = st.number_input("Gula (g)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    
    if st.button("Tambah Asupan Makanan", key="add_food"):
        food_cal = calculate_food_calories(f_fat, f_carb, f_protein, f_sugars)
        st.session_state.total_calories_consumed += food_cal
        st.success(f"Total Kalori **{food_cal:.0f} kkal**.")

    st.markdown("---")

    # Tombol Reset
    if st.button("Reset Laporan Harian"):
        st.session_state.total_calories_consumed = 0
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)


# Mengisi placeholder laporan dengan data terbaru
with report_placeholder.container():
    calories_left = daily_goal - st.session_state.total_calories_consumed
    progress_percent = (st.session_state.total_calories_consumed / daily_goal) * 100
    progress_angle = max(0, min(360, progress_percent * 3.6))

    st.markdown(f"""
    <div class="progress-ring" style="--progress-angle: {progress_angle}deg;">
        <div class="progress-text">
            <h2>{calories_left:.0f}</h2>
            <p>kkal Tersisa</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.total_calories_consumed == 0:
        st.info("Ayo mulai lacak asupan kalori Anda hari ini!")
    elif calories_left > 0:
        st.success(f"Kerja bagus! Anda masih punya **{calories_left:.0f} kkal** untuk mencapai target.")
    else:
        st.warning(f"Target tercapai! Anda surplus **{-calories_left:.0f} kkal**.")
