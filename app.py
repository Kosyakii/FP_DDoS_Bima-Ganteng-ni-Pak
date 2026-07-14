import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import warnings

# Menyembunyikan peringatan versi XGBoost
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Konfigurasi Halaman
st.set_page_config(page_title="DDoS Detection System", page_icon="🛡️", layout="wide")

# Menggunakan cache_resource untuk Load Model ML
@st.cache_resource
def load_assets():
    model = joblib.load('best_model_ddos.pkl')
    scaler = joblib.load('scaler_ddos.pkl')
    return model, scaler

model, scaler = load_assets()
expected_features = scaler.feature_names_in_

# ==========================================
# 2. INISIALISASI MEMORI TERPADU (SESSION STATE)
# ==========================================
if 'df_working' not in st.session_state:
    st.session_state['df_working'] = pd.DataFrame(columns=expected_features)

if 'df_result' not in st.session_state:
    st.session_state['df_result'] = None

if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

# ==========================================
# 3. SIDEBAR (UPLOAD & FORM INPUT)
# ==========================================
with st.sidebar:
    st.header("📂 1. Upload File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    # Logika Cerdas Memuat File
    if uploaded_file is not None:
        if st.session_state['last_uploaded_file'] != uploaded_file.name:
            st.session_state['df_working'] = pd.read_csv(uploaded_file)
            st.session_state['last_uploaded_file'] = uploaded_file.name
            st.session_state['df_result'] = None 
    else:
        if st.session_state['last_uploaded_file'] is not None:
            st.session_state['df_working'] = pd.DataFrame(columns=expected_features)
            st.session_state['last_uploaded_file'] = None
            st.session_state['df_result'] = None
            
    st.markdown("---")
    
    st.header("📝 2. Input Data Baru")
    st.markdown("Isi form ini untuk Menambahkan Data baru kedalam list data.")
    
    with st.form("manual_input_form"):
        sidebar_cols = st.columns(2)
        input_data = {}
        
        for i, feature in enumerate(expected_features):
            col = sidebar_cols[i % 2]
            # MENGUBAH KE TEXT_INPUT: Bisa diisi huruf dan menghilangkan format 0.00
            input_data[feature] = col.text_input(label=feature, value="0")
            
        st.markdown(" ")
        submitted = st.form_submit_button("➕ Tambahkan ke Tabel", use_container_width=True)
        
        if submitted:
            new_row = pd.DataFrame([input_data])
            
            if st.session_state['df_working'].empty:
                st.session_state['df_working'] = new_row
            else:
                st.session_state['df_working'] = pd.concat([st.session_state['df_working'], new_row], ignore_index=True)
            
            st.session_state['df_result'] = None
            st.success("Berhasil ditambahkan ke baris paling bawah!")

# ==========================================
# 4. HALAMAN UTAMA (MENAMPILKAN & MEMPROSES DATA)
# ==========================================
st.title("🛡️ Sistem Deteksi Serangan DDoS")
st.markdown("Unggah file CSV atau tambahkan data dari sidebar. Anda juga bisa mengedit angka langsung di tabel ini sebelum mendeteksi.")

if not st.session_state['df_working'].empty:
    
    st.subheader("📝 Pratinjau Dataset Gabungan")
    
    edited_df = st.data_editor(st.session_state['df_working'], num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 Jalankan Deteksi DDoS", use_container_width=True):
        with st.spinner('Menganalisis seluruh baris data...'):
            # Menyiapkan kolom sesuai model
            df_processed = edited_df.reindex(columns=expected_features, fill_value=0)
            
            # PENGAMANAN ALFABET: Memaksa semua data huruf diubah menjadi angka 0 agar model tidak crash
            df_processed = df_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Proses Machine Learning
            df_scaled = scaler.transform(df_processed)
            predictions = model.predict(df_scaled)
            
            # Menyimpan hasil ke tabel untuk ditampilkan
            df_result = edited_df.copy()
            df_result['Prediction'] = predictions
            df_result['Status'] = df_result['Prediction'].apply(lambda x: "🚨 DDoS Detected" if x == 1 else "✅ Normal")
            
            cols = ['Status'] + [col for col in df_result.columns if col not in ['Status', 'Prediction']]
            df_result = df_result[cols]
            
            st.session_state['df_result'] = df_result

else:
    st.info("👋 Belum ada data yang diproses. Silakan upload file CSV atau klik 'Tambahkan ke Tabel' di sidebar kiri.")

# ==========================================
# 5. MENAMPILKAN HASIL DETEKSI (JIKA ADA)
# ==========================================
if st.session_state['df_result'] is not None:
    df_final = st.session_state['df_result']
    st.success("Deteksi Selesai!")
    
    total_traffic = len(df_final)
    total_ddos = sum(df_final['Status'] == "🚨 DDoS Detected")
    total_normal = sum(df_final['Status'] == "✅ Normal")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Dianalisis", total_traffic)
    col2.metric("Aktivitas Normal", total_normal)
    col3.metric("Indikasi DDoS", total_ddos, delta_color="inverse")
    
    st.subheader("🔍 Hasil Deteksi Akhir")
    st.dataframe(df_final, use_container_width=True) 
    
    csv_output = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Hasil Deteksi",
        data=csv_output,
        file_name='ddos_detection_results.csv',
        mime='text/csv',
    )