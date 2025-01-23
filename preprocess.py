from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    # Membuka gambar
    img = Image.open(image_path)

    # Jika gambar grayscale (1 saluran), ubah menjadi RGB (3 saluran)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Ubah ukuran gambar dan normalisasi
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0

    # Menambah dimensi untuk batch size (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


