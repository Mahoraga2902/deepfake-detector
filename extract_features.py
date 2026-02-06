import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from glob import glob
import argparse
from tqdm import tqdm

# ---------- CONFIG ----------
MODEL_PATH = "deepfake_model.h5"     # CNN model in your project root
VIDEO_DIR = "videos"                 # expects videos/train/ and videos/val/
OUTPUT_DIR = "features"
IMG_SIZE = (128, 128)                # adjust if your CNN uses a different size
SAMPLE_EVERY_SECONDS = 0.5
MAX_FRAMES = 64
# ----------------------------

def build_feature_extractor(model_path):
    model = load_model(model_path)
    try:
        features_out = model.layers[-2].output
        extractor = Model(inputs=model.input, outputs=features_out)
        print("[INFO] Feature extractor: Using layers[-2]")
    except Exception as e:
        print("[WARN] Could not use penultimate layer. Using entire model:", e)
        extractor = model
    return extractor

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

def extract_video_features(video_path, extractor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, extractor.output_shape[-1]), dtype=np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = int(max(fps * SAMPLE_EVERY_SECONDS, 1))

    frames = []
    idx = 0
    while len(frames) < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
        idx += stride

    cap.release()

    if len(frames) == 0:
        return np.zeros((0, extractor.output_shape[-1]), dtype=np.float32)

    frames = np.array(frames)
    features = extractor.predict(frames, verbose=0)
    return features.astype(np.float32)

def ensure(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    extractor = build_feature_extractor(MODEL_PATH)

    for split in ["train", "val"]:
        video_path_root = os.path.join(VIDEO_DIR, split)
        out_root = os.path.join(OUTPUT_DIR, split)

        ensure(out_root)

        video_files = glob(os.path.join(video_path_root, "*.mp4"))
        print(f"[INFO] Found {len(video_files)} videos in {split}")

        for vp in tqdm(video_files, desc=f"Extracting {split}"):
            vid_name = os.path.splitext(os.path.basename(vp))[0]
            save_path = os.path.join(out_root, vid_name + ".npy")

            if os.path.exists(save_path):
                continue

            feats = extract_video_features(vp, extractor)
            np.save(save_path, feats)

if __name__ == "__main__":
    main()
