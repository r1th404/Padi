import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================
# Load TFLite Model
# ============================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# ============================
# Prediksi dengan TFLite
# ============================
def predict_tflite(img):
    img = img.resize((224, 224))            # sesuaikan dengan model kamu
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Label sesuai model kamu
LABELS = ["Healthy", "Brown Spot", "Leaf Blast", "Hispa"]

# ============================
# UI Aplikasi
# ============================
st.title("🌾 PadiSehat – Deteksi Penyakit Daun Padi (TFLite Version)")
st.write("Upload foto daun padi untuk mendeteksi penyakit otomatis.")

uploaded_img = st.file_uploader("Upload gambar daun padi...", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    if st.button("Deteksi Penyakit"):
        preds = predict_tflite(img)
        idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)

        st.subheader("Hasil Deteksi")
        st.success(f"Penyakit: **{LABELS[idx]}**")
        st.info(f"Akurasi Model: {confidence:.2f}%")
