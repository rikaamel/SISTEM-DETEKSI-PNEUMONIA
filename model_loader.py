import tensorflow as tf

def load_model(model_path="cnn_model.h5"):
    """
    Memuat model CNN yang sudah dilatih.
    """
    return tf.keras.models.load_model(model_path)
