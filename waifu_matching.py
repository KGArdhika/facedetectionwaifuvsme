import cv2
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load model deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ Tidak ada wajah terdeteksi: {image_path}")
        return None

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (100, 100))
    features = face_resized.flatten() / 255.0
    return features

def process_waifu_dataset(dataset_path="waifu_dataset"):
    cache_path = "waifu_features_cache.json"
    if os.path.exists(cache_path):
        print("🔁 Memuat fitur dari cache...")
        with open(cache_path, "r") as f:
            waifu_features = json.load(f)
        return waifu_features

    waifu_features = {}
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(dataset_path, filename)
            features = extract_face_features(img_path)
            if features is not None:
                waifu_features[filename] = features.tolist()

    with open(cache_path, "w") as f:
        json.dump(waifu_features, f)

    print(f"✅ Total waifu yang diproses: {len(waifu_features)}")
    return waifu_features

def find_best_match(user_features, waifu_features):
    best_match = None
    best_score = -1

    for name, features in waifu_features.items():
        similarity = cosine_similarity([user_features], [features])[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = name

    return best_match, best_score

# === MAIN ===
name = input("Masukkan nama Anda: ")
user_img_path = input("Masukkan path ke gambar Anda: ")

user_features = extract_face_features(user_img_path)
if user_features is None:
    print("❌ Gagal mengekstrak fitur wajah.")
    exit()

waifu_features = process_waifu_dataset()

print("🔎 Mencari pasangan waifu terbaik...")
match_name, score = find_best_match(user_features, waifu_features)

if match_name:
    print(f"💘 Pasangan waifu terbaik untuk {name} adalah: {match_name}")
    print(f"📊 Skor kemiripan: {score:.4f}")
else:
    print("❌ Tidak ditemukan pasangan yang cocok.")
