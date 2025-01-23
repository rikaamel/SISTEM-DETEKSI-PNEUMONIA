from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Memuat model yang sudah dilatih
model = load_model('cnn_model.h5')

# Fungsi untuk memproses gambar
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Menggunakan model untuk prediksi
image = preprocess_image('person1_virus_6.jpeg')
prediction = model.predict(image)

# Menampilkan hasil prediksi
if prediction[0][0] > 0.5:
    print("Pneumonia")
else:
    print("Normal")
