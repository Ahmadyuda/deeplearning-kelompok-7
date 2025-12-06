import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# --- Konfigurasi Global ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'mobilenetv2_lettuce_best_finetuned.keras'
CLASS_NAMES = ['Sehat', 'Sakit'] # Nama kelas dalam Bahasa Indonesia

# --- Fungsi Pemuatan Model (Menggunakan Cache Streamlit) ---
# @st.cache_resource memastikan model hanya dimuat sekali
@st.cache_resource
def load_my_model():
    try:
        # Pemuatan dengan custom_objects untuk metrik
        model = load_model(
            MODEL_PATH, 
            custom_objects={
                'Precision': tf.keras.metrics.Precision, 
                'Recall': tf.keras.metrics.Recall
            }
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# --- Fungsi Preprocessing ---
def preprocess_image(image):
    # 1. Resize Gambar ke Ukuran Input Model (224x224)
    img = image.resize(IMG_SIZE)
    
    # 2. Konversi ke Array NumPy dan tambahkan dimensi batch
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    
    # 3. Preprocessing MobileNetV2 (Rescaling ke [-1, 1])
    # Menggunakan tf.cast karena input MobileNetV2 memerlukan float32
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
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
        
        # Tombol Prediksi
        if st.button('Prediksi'):
            with st.spinner('Menganalisis gambar...'):
                
                # 1. Preprocessing
                processed_input = preprocess_image(image)
                
                # 2. Prediksi
                prediction = model.predict(processed_input)
                
                # Probabilitas kelas 'Sakit'
                unhealthy_prob = prediction[0][0] 
                
                # 3. Klasifikasi
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
