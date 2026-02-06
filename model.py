# model.py (UPDATED to return frame image)
import os
import numpy as np
import cv2
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "deepfake_model.h5")
DEFAULT_KERAS = "deepfake_transformer.keras"
DEFAULT_H5 = "deepfake_transformer.h5"
TRANSFORMER_MODEL_PATH = os.environ.get("TRANSFORMER_MODEL_PATH",
                                       DEFAULT_KERAS if os.path.exists(DEFAULT_KERAS) else DEFAULT_H5)
IMG_SIZE = (128, 128)
SAMPLE_EVERY_SECONDS = 0.5
MAX_FRAMES = 64
SEQ_LEN = 48
THRESHOLD = 0.5
# ----------------------------------------

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, dtype=tf.int32)
        pos_emb = self.position_embedding(positions)
        return x + pos_emb
    def get_config(self):
        return {"max_len": self.max_len, "embed_dim": self.embed_dim}

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = None
        self.ffn = None
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def build(self, input_shape):
        key_dim = max(1, self.embed_dim // max(1, self.num_heads))
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim, dropout=self.rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu'),
            layers.Dense(self.embed_dim)
        ])
        super().build(input_shape)
    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}

print("[model.py] Loading CNN and Transformer models...")
if not os.path.exists(CNN_MODEL_PATH):
    print(f"[model.py] WARNING: CNN model not found at {CNN_MODEL_PATH}")
cnn = load_model(CNN_MODEL_PATH)
try:
    feat_extractor = Model(inputs=cnn.input, outputs=cnn.layers[-2].output)
    print("[model.py] Using penultimate layer as feature extractor.")
except Exception as e:
    feat_extractor = cnn
    print("[model.py] Could not use penultimate layer; using full model as extractor:", e)

if not os.path.exists(TRANSFORMER_MODEL_PATH):
    print(f"[model.py] WARNING: Transformer model not found at {TRANSFORMER_MODEL_PATH}")

try:
    transformer = load_model(TRANSFORMER_MODEL_PATH)
    print("[model.py] Transformer loaded (no custom_objects).")
except Exception as e_no:
    print("[model.py] Warning: direct load failed, retrying with custom_objects:", str(e_no))
    transformer = load_model(TRANSFORMER_MODEL_PATH, custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerBlock": TransformerBlock
    })
    print("[model.py] Transformer loaded (with custom_objects).")

print("[model.py] Models loaded.")

def preprocess_frame_bgr(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

def frames_to_features(video_path, sample_every_sec=SAMPLE_EVERY_SECONDS, max_frames=MAX_FRAMES):
    """
    Returns (features, frames_rgb_uint8) or (None, None, error_str)
    - features: numpy array (T,D)
    - frames_rgb_uint8: list or array of uint8 RGB frames (shape (T, H, W, 3))
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "cannot_open_video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps * sample_every_sec)))
    frames = []
    frames_uint8 = []
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames_uint8.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # keep original RGB uint8
        frames.append(preprocess_frame_bgr(frame))
        idx += stride
    cap.release()
    if len(frames) == 0:
        return None, None, "no_frames_extracted"
    frames_arr = np.stack(frames, axis=0)
    feats = feat_extractor.predict(frames_arr, verbose=0)
    return feats.astype(np.float32), np.stack(frames_uint8, axis=0), None

def pad_or_truncate(feats, seq_len=SEQ_LEN):
    t = feats.shape[0]
    if t >= seq_len:
        return feats[:seq_len]
    pad_len = seq_len - t
    return np.pad(feats, ((0,pad_len),(0,0)), mode='constant', constant_values=0.0)

def _frame_to_base64_jpeg(frame_rgb_uint8, jpeg_quality=85):
    # frame_rgb_uint8: H,W,3 in RGB uint8
    bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return b64

def predict_video_from_path(video_path):
    """
    Returns dict with keys:
    - fake_prob: float
    - label: 'fake'|'real'
    - frame_base64: (optional) JPEG base64 string of representative frame
    Or returns {'error':...}
    """
    feats, frames_rgb, err = frames_to_features(video_path)
    if feats is None:
        return {"error": err}
    seq = pad_or_truncate(feats, SEQ_LEN)
    x = np.expand_dims(seq, axis=0).astype(np.float32)
    try:
        prob = float(transformer.predict(x, verbose=0)[0,0])
    except Exception as e:
        return {"error": "inference_error", "detail": str(e)}

    # pick a representative frame: choose middle of sampled frames (clamped to seq_len-1)
    # frames_rgb shape may be smaller than seq_len; use frames_rgb
    try:
        T = frames_rgb.shape[0]
        mid = T // 2
        # if frames were padded/truncated we still have frames_rgb length = original sampled frames
        frame_rgb = frames_rgb[mid]
        b64 = _frame_to_base64_jpeg(frame_rgb)
    except Exception:
        b64 = None

    label = "fake" if prob >= THRESHOLD else "real"
    out = {"fake_prob": prob, "label": label}
    if b64 is not None:
        out["frame_base64"] = b64
    return out

def predict_features_array(feats_array):
    seq = pad_or_truncate(feats_array, SEQ_LEN)
    x = np.expand_dims(seq, axis=0).astype(np.float32)
    prob = float(transformer.predict(x, verbose=0)[0,0])
    label = "fake" if prob >= THRESHOLD else "real"
    return {"fake_prob": prob, "label": label}
