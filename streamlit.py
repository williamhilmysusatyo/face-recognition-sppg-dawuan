import streamlit as st
import cv2
import numpy as np
import insightface
from PIL import Image
import tempfile
import os
import gspread
import base64
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pandas as pd

# === Setup Credential & Connect ke Google Sheets ===
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
base64_creds = st.secrets["gcp"]["base64_creds"]

# Decode dan konversi ke dict
creds_dict = json.loads(base64.b64decode(base64_creds).decode("utf-8"))

creds = ServiceAccountCredentials.from_json_keyfile_name(creds_dict, scope)
client = gspread.authorize(creds)

# # === Buka Spreadsheet ===
absensi = client.open("Database Karyawan SPPG Dawuan").worksheet("Absensi")

@st.cache_resource
def load_model_and_refs(ref_dir="image"):
    model = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0)

    known_embeddings = []
    known_names = []

    for filename in os.listdir(ref_dir):
        if filename.endswith(('.jpg', '.png', '.JPG')):
            img_path = os.path.join(ref_dir, filename)
            img = cv2.imread(img_path)
            faces = model.get(img)
            if faces:
                known_embeddings.append(faces[0].embedding)
                known_names.append(os.path.splitext(filename)[0])
    return model, known_embeddings, known_names

def recognize_face(model, known_embeddings, known_names, img, threshold=0.55):
    faces = model.get(img)
    if not faces:
        return "Tidak ada wajah terdeteksi", img

    face = faces[0]
    emb = face.embedding

    # Hitung similarity dengan semua wajah referensi
    sims = [
            np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            for known_emb in known_embeddings
        ]

    best_match_idx = np.argmax(sims)
    best_score = sims[best_match_idx]

    if best_score < threshold:
        return "Tidak teridentifikasi", img

    name = known_names[best_match_idx]
    similarity_percentage = best_score * 100
    similarity_text = f"{name} ({similarity_percentage:.1f}%)"

    # Draw bounding box dan nama+similarity
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, similarity_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return name, img

st.title("Face Recognition from Webcam")
model, known_embeddings, known_names = load_model_and_refs()

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    # Convert to cv2 image
    img = Image.open(camera_image)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    name, annotated_img = recognize_face(model, known_embeddings, known_names, img)

    st.image(annotated_img, channels="BGR", caption=f"Recognized: {name}")
    absensi_data = absensi.get_all_values()
    df_absensi = pd.DataFrame(absensi_data[1:], columns=absensi_data[0])  # Skip header
    
    
    if name != "Tidak ada wajah terdeteksi" and name != "Tidak teridentifikasi":
        st.success(f"Wajah dikenali sebagai: {name}")
        now = datetime.now()
        tanggal = now.strftime("%Y-%m-%d")
        waktu = now.strftime("%H:%M:%S")
        user_today = df_absensi[
            (df_absensi["Nama"] == name) & (df_absensi["Tanggal"] == tanggal)
            ]

        # Tentukan status
        if user_today.empty:
            status = "Masuk"
        else:
            last_status = user_today.iloc[-1]["Status"]
            status = "Keluar" if last_status == "Masuk" else "Masuk"
            
        absensi.append_row([tanggal, waktu, name, "Posisi", status])
        st.info("Absensi berhasil dicatat.")
