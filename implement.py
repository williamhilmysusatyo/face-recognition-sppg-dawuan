import insightface
import cv2
import numpy as np
import time
import os

# === Load Model ===
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)

# === Load Semua Gambar Referensi ===
ref_dir = "image"
known_embeddings = []
known_names = []

for filename in os.listdir(ref_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(ref_dir, filename)
        image = cv2.imread(path)
        faces = model.get(image)

        if faces:
            known_embeddings.append(faces[0].embedding)
            name = os.path.splitext(filename)[0]
            known_names.append(name)
        else:
            print(f"[!] Wajah tidak ditemukan di: {filename}")

# === Threshold & Timer ===
threshold = 0.35
timeout_seconds = 10
start_time = time.time()
matched = False
matched_name = "Unknown"

# === Buka Kamera ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca dari kamera.")
        break

    faces = model.get(frame)

    for face in faces:
        emb = face.embedding
        box = face.bbox.astype(int)

        # Hitung cosine similarity ke semua known embeddings
        similarities = [
            np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            for known_emb in known_embeddings
        ]

        # Cari similarity tertinggi
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score > threshold:
            matched = True
            matched_name = known_names[best_idx]
            print(f"[MATCH] {matched_name} (similarity: {best_score:.4f})")
            break  # Langsung keluar dari loop wajah

    # Tampilkan frame (optional)
    cv2.imshow("Face Recognition", frame)

    if (time.time() - start_time) > timeout_seconds or matched:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Bersihkan ===
cap.release()
cv2.destroyAllWindows()

# === Output Hasil ===
if matched:
    print(f"✅ Wajah dikenali sebagai: {matched_name}. Kamera ditutup.")
else:
    print("⏰ Timeout: Tidak ada wajah yang cocok ditemukan.")
