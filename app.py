import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# --- Konfigurasi Global ---
IMG_SIZE = (224, 224)
# GANTI NAMA FILE MODEL KARENA KONFLIK VERSI KERAS
MODEL_PATH = 'mobilenetv2_legacy_h5.h5' 
CLASS_NAMES = ['Sehat', 'Sakit'] 

# --- Fungsi Pemuatan Model (Menggunakan Cache Streamlit) ---
@st.cache_resource
def load_my_model():
    # SETELAN CPU: Memaksa TensorFlow hanya menggunakan CPU untuk menghindari error GPU di Docker
    tf.config.set_visible_devices([], 'GPU')
    
    try:
        # Pemuatan model H5 lama (lebih stabil)
        # Format H5 lama umumnya tidak memerlukan custom_objects untuk metrik standar
        model = load_model(MODEL_PATH) 
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: Pastikan file '{MODEL_PATH}' ada di root direktori. Error: {e}")
        st.stop()

# --- Fungsi Preprocessing ---
def preprocess_image(image):
    # 1. Resize Gambar
    img = image.resize(IMG_SIZE)
    
    # 2. Konversi ke Array NumPy (float32) dan tambahkan dimensi batch
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # 3. Preprocessing MobileNetV2 (Rescaling ke [-1, 1])
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return processed_img

# --- Fungsi Utama Aplikasi Streamlit ---
def main():
    st.title("🌱 Deteksi Kesehatan Selada Otomatis")
    st.markdown("Unggah foto daun selada untuk memprediksi apakah ia **Sehat** atau **Sakit**.")

    # Muat Model
    model = load_my_model()

    # Widget Upload File
    uploaded_file = st.file_uploader("Pilih gambar daun selada...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
        
        if st.button('Prediksi'):
            with st.spinner('Menganalisis gambar...'):
                
                # 1. Preprocessing
                processed_input = preprocess_image(image)
                
                # 2. Prediksi
                prediction = model.predict(processed_input)
                
                # Probabilitas kelas 'Sakit' (Index 1)
                unhealthy_prob = prediction[0][0] 
                
                # 3. Klasifikasi (Threshold 0.5)
                if unhealthy_prob >= 0.5:
                    predicted_class = CLASS_NAMES[1] # Sakit
                    confidence = unhealthy_prob
                else:
                    predicted_class = CLASS_NAMES[0] # Sehat
                    confidence = 1.0 - unhealthy_prob

                # 4. Tampilkan Hasil
                st.markdown("---")
                if predicted_class == 'Sakit':
                    st.error(f"⚠️ Hasil: **{predicted_class}**")
                else:
                    st.success(f"✅ Hasil: **{predicted_class}**")
                    
                st.info(f"Keyakinan ({predicted_class}): {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
