import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Memuat model yang telah disimpan
model = load_model('cnn_model.h5')  # Ganti dengan path model yang disimpan

# Menyiapkan data generator untuk evaluasi (test data)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',  # Ganti dengan folder data uji yang sesuai
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',  # Klasifikasi biner
    color_mode='rgb'  # Pastikan gambar dimuat dengan 3 saluran warna
)

# Evaluasi model pada data uji
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Mengambil prediksi dari model untuk semua data
y_true = test_generator.classes  # Label asli dari data uji
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)

# Menentukan label prediksi (0 atau 1) untuk klasifikasi biner
y_pred = (y_pred > 0.5).astype(int)  # Threshold 0.5 untuk sigmoid

# Memastikan jumlah data prediksi sesuai dengan jumlah data asli
print(f"Jumlah data asli (y_true): {len(y_true)}")
print(f"Jumlah data prediksi (y_pred): {len(y_pred)}")

# Menghitung Confusion Matrix secara manual
tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

# Menampilkan Confusion Matrix secara manual
cm = [[tn, fp], [fn, tp]]

# Visualisasi confusion matrix dengan matplotlib
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

# Labeling matrix
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticklabels(['', 'Negatif', 'Positif'])
ax.set_yticklabels(['', 'Negatif', 'Positif'])

# Menambahkan angka pada setiap kotak di confusion matrix
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

plt.title('Confusion Matrix')
plt.show()

# Menghitung metrik lainnya (Precision, Recall, F1-Score)
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")
