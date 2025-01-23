import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model('cnn_model.h5')

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(128, 128)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch size
    return img_array

# Judul aplikasi
st.title("Deteksi Penyakit dengan Citra Medis")

# Upload gambar untuk prediksi
uploaded_image = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Menampilkan gambar yang diupload
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    # Memproses gambar untuk prediksi
    img_array = preprocess_image(img)

    # Melakukan prediksi
    prediction = model.predict(img_array)

    # Menampilkan hasil prediksi
    if prediction[0][0] > 0.5:
        st.write("**Prediksi: Pneumonia**")
    else:
        st.write("**Prediksi: Normal**")
