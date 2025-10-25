import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
import cv2

MODEL_PATH = "deepfake_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128)
CLASS_NAMES = ["real", "fake"]

def _preprocess_bgr_frame(frame_bgr):
    img = cv2.resize(frame_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)

        if predictions.shape[-1] == 1:
            p_fake = float(predictions[0][0])
            label = "fake" if p_fake >= 0.5 else "real"
            confidence = p_fake if label == "fake" else 1 - p_fake
        else:
            p = predictions[0]
            idx = int(np.argmax(p))
            label = CLASS_NAMES[idx]
            confidence = float(np.max(p))

        return {"label": label, "confidence": round(confidence, 3)}
    except Exception as e:
        return {"error": str(e)}

def predict_video(video_path, sample_every_sec=1.0, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0
    stride = max(int(round(fps * sample_every_sec)), 1)

    probs_fake = []
    frames_sampled = 0
    frame_idx = 0
    grabbed = True

    while grabbed and frames_sampled < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        grabbed, frame = cap.read()
        if not grabbed:
            break

        arr = _preprocess_bgr_frame(frame)
        preds = model.predict(arr, verbose=0)

        if preds.shape[-1] == 1:
            p_fake = float(preds[0][0])
        else:
            p_fake = float(preds[0][1]) if len(preds[0]) >= 2 else float(preds[0][0])

        probs_fake.append(p_fake)
        frames_sampled += 1
        frame_idx += stride

    cap.release()

    if frames_sampled == 0:
        return {"error": "No frames sampled from video."}

    mean_p_fake = float(np.mean(probs_fake))
    label = "fake" if mean_p_fake >= 0.5 else "real"
    confidence = mean_p_fake if label == "fake" else 1 - mean_p_fake

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "frames_sampled": frames_sampled,
        "video_fps": round(float(fps), 2),
        "estimated_duration_sec": round(float(duration), 2)
    }
