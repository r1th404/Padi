import streamlit as st
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

# ======================================================
# LOAD METADATA
# ======================================================
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

class_indices = metadata["class_indices"]
class_names = metadata["class_names"]
label_map = metadata["label_map"]
input_h, input_w, _ = metadata["input_shape"]

# Reverse class_indices → index_to_label
index_to_label = {v: k for k, v in class_indices.items()}

# ======================================================
# LOAD ONNX MODEL
# ======================================================
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(
    page_title="PadiSehat - Deteksi Penyakit Padi",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 PadiSehat — Deteksi Penyakit Daun Padi")
st.write("Upload gambar daun padi untuk mendeteksi penyakit secara otomatis.")

uploaded_file = st.file_uploader("Upload gambar daun padi", type=["jpg", "jpeg", "png"])

# ======================================================
# PREDICT FUNCTION
# ======================================================
def preprocess_image(image):
    image = image.resize((input_w, input_h))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_array):
    outputs = session.run(None, {input_name: img_array})
    preds = outputs[0][0]
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    return class_id, confidence

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Terupload", use_column_width=True)

    if st.button("🔍 Deteksi Penyakit"):
        with st.spinner("Sedang menganalisis gambar..."):
            img_array = preprocess_image(image)
            class_id, confidence = predict(img_array)

            eng_label = index_to_label[class_id]
            human_label = label_map[eng_label]

            st.success(f"**Hasil Deteksi: {human_label}**")
            st.write(f"Confidence: **{confidence*100:.2f}%**")

            st.info(f"Label model: `{eng_label}`")

        st.write("---")
        st.caption("Model: MobileNetV2 • ONNX Runtime")
