import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import warnings

# Menyembunyikan peringatan versi XGBoost
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Konfigurasi Halaman
st.set_page_config(page_title="DDoS Detection System", page_icon="🛡️", layout="wide")

# Menggunakan cache versi lama yang kompatibel dengan sistem Anda
@st.cache(allow_output_mutation=True)
def load_assets():
    model = joblib.load('best_model_ddos.pkl')
    scaler = joblib.load('scaler_ddos.pkl')
    return model, scaler

model, scaler = load_assets()
expected_features = scaler.feature_names_in_

# ==========================================
# 2. INISIALISASI MEMORI (SESSION STATE)
# ==========================================
# Ini wajib ada di luar blok proses agar tidak kerestart
if 'df_result' not in st.session_state:
    st.session_state['df_result'] = None

# Fungsi ini akan otomatis dipanggil untuk mereset memori jika file baru diupload
def clear_results():
    st.session_state['df_result'] = None

# 3. Antarmuka Utama
st.title("🛡️ Sistem Deteksi Serangan DDoS")
st.markdown("Unggah log jaringan Anda. **Hasil akan terus tampil di layar** sampai Anda mengunggah file baru.")

# 4. Sidebar Input
with st.sidebar:
    st.header("📂 Upload Data Jaringan")
    # Parameter on_change=clear_results menyambung ke fungsi reset di atas
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'], on_change=clear_results)

# 5. Proses Utama
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Pratinjau Data Mentah")
        st.dataframe(df_raw.head())

        # ==========================================
        # BLOK TOMBOL: HANYA UNTUK HITUNG & SIMPAN
        # ==========================================
        if st.button("🚀 Jalankan Deteksi DDoS"):
            with st.spinner('Menyelaraskan fitur dan menganalisis traffic...'):
                df_processed = df_raw.reindex(columns=expected_features, fill_value=0)
                df_scaled = scaler.transform(df_processed)
                predictions = model.predict(df_scaled)
                
                df_result = df_raw.copy()
                df_result['Prediction'] = predictions
                df_result['Status'] = df_result['Prediction'].apply(lambda x: "🚨 DDoS Detected" if x == 1 else "✅ Normal")
                
                cols = ['Status'] + [col for col in df_result.columns if col not in ['Status', 'Prediction']]
                df_result = df_result[cols]
                
                # SIMPAN HASIL KE MEMORI! 
                st.session_state['df_result'] = df_result

        # ==========================================
        # 6. MENAMPILKAN HASIL DARI MEMORI
        # ==========================================
        # Pastikan blok ini berada SEJAJAR dengan `if st.button`, BUKAN di dalamnya!
        if st.session_state['df_result'] is not None:
            df_final = st.session_state['df_result']
            st.success("Deteksi Selesai!")
            
            # Hitung metrik
            total_traffic = len(df_final)
            total_ddos = sum(df_final['Status'] == "🚨 DDoS Detected")
            total_normal = sum(df_final['Status'] == "✅ Normal")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Traffic (Baris)", total_traffic)
            col2.metric("Aktivitas Normal", total_normal)
            col3.metric("Indikasi DDoS", total_ddos, delta_color="inverse")
            
            st.subheader("🔍 Hasil Deteksi Akhir")
            st.dataframe(df_final) 
            
            # Opsi Download
            csv_output = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Deteksi",
                data=csv_output,
                file_name='ddos_detection_results.csv',
                mime='text/csv',
            )
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.warning("Menunggu unggahan file CSV di menu sebelah kiri.")