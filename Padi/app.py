import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# ==============================
# LOAD MODEL & METADATA
# ==============================
MODEL_PATH = "best_model_mobilenetv2.keras"
META_PATH = "model_metadata.json"

model = load_model(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

class_indices = metadata["class_indices"]
class_names = metadata["class_names"]
label_map = metadata["label_map"]

IMG_SIZE = (224, 224)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img


# ==============================
#   UI STREAMLIT
# ==============================
st.set_page_config(page_title="PadiSehat", layout="wide")

st.title("🌾 PadiSehat – Sistem Deteksi Penyakit Padi")
menu = st.sidebar.radio("Navigasi", ["Deteksi Penyakit", "Dashboard", "Logbook", "Informasi"])


# ==============================
# 1. DETEKSI PENYAKIT
# ==============================
if menu == "Deteksi Penyakit":
    st.header("🔍 Deteksi Penyakit Otomatis")
    st.write("Upload foto daun padi untuk mendeteksi penyakit secara instan.")

    uploaded_file = st.file_uploader("Upload Foto Daun", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Preview Gambar")
            st.image(img, width=300)

        with col2:
            st.subheader("Hasil Prediksi")
            img_array = preprocess_image(img)
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100

            class_name = class_names[str(pred_idx)]
            prediction_id = label_map[class_name]

            st.success(f"📌 **{prediction_id}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.caption("Model: MobileNetV2 (224×224)")

        # Rekomendasi otomatis
        rekomendasi = {
            "Hawar Daun Bakteri": "Gunakan varietas tahan dan perbaiki sanitasi lahan.",
            "Bercak Cokelat": "Gunakan fungisida & pupuk berimbang.",
            "Blast Daun": "Gunakan jarak tanam ideal untuk mengurangi kelembaban.",
            "Hispa": "Gunakan perangkap atau insektisida jika diperlukan.",
        }

        st.subheader("Rekomendasi Penanganan")
        st.info(rekomendasi.get(prediction_id, "Tidak ada rekomendasi khusus."))

# ==============================
# 2. DASHBOARD
# ==============================
elif menu == "Dashboard":
    st.header("📊 Dashboard Monitoring")
    st.write("Pantauan cepat kondisi penyakit di berbagai wilayah.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BLB Terlapor", "45 kasus")
    col2.metric("Bercak Cokelat", "28 kasus")
    col3.metric("Blast Daun", "19 kasus")
    col4.metric("Hispa", "8 kasus")

# ==============================
# 3. LOGBOOK
# ==============================
elif menu == "Logbook":
    st.header("📘 Logbook Perawatan Sawah")
    st.write("Catatan aktivitas dan kondisi lahan.")

    data = [
        "01/11/2025 - Sawah A: BLB ringan, penyemprotan fungisida",
        "03/11/2025 - Sawah B: Bercak Cokelat, pemupukan tambahan",
        "05/11/2025 - Sawah C: Blast Daun, monitoring intensif",
    ]

    for item in data:
        st.info(item)

# ==============================
# 4. INFORMASI
# ==============================
elif menu == "Informasi":
    st.header("📰 Informasi & Berita Terkini")
    st.write("Berbagi informasi dan diskusi antar petani.")

    st.subheader("Lapor Cepat")
    nama = st.text_input("Nama/Asal (opsional)")
    laporan = st.text_area("Tulis laporan atau pertanyaan")

    if st.button("Kirim"):
        st.success("Laporan Anda berhasil dikirim!")

    st.subheader("🔥 Forum Diskusi")
    st.info("Pak Widodo (Desa Sukamaju): Ada bercak cokelat di sawah blok B?")
    st.info("Penyuluh Dinas Pertanian: Rotasi tanaman & sanitasi lahan.")
