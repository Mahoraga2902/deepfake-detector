# evaluate_transformer.py
# Put this in your project root and run: python evaluate_transformer.py

import os, glob, numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

MODEL_PATH = "deepfake_transformer.h5"
FEATURE_DIR_VAL = "features/val"
SEQ_LEN = 48

# --- Recreate the custom layers used when training ---
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        # keep key_dim consistent with how the model was built (embed_dim // num_heads)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)  # will be re-bound below
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        # store config (Keras will call get_config/from_config when deserializing)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}

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

# --- Load model with custom objects ---
custom_objects = {
    "TransformerBlock": TransformerBlock,
    "PositionalEmbedding": PositionalEmbedding
}

print("[INFO] Loading model (may warn about oneDNN; ignore)...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print("[INFO] Model loaded.")

# --- Helper: pad/truncate ---
def load_and_prepare(path, seq_len=SEQ_LEN):
    arr = np.load(path)
    t, d = arr.shape
    if t >= seq_len:
        arr2 = arr[:seq_len]
    else:
        pad = seq_len - t
        arr2 = np.pad(arr, ((0,pad),(0,0)), mode='constant', constant_values=0.0)
    return arr2

# --- Gather val files ---
files = sorted(glob.glob(os.path.join(FEATURE_DIR_VAL, "*.npy")))
if len(files) == 0:
    raise SystemExit(f"No files in {FEATURE_DIR_VAL}. Make sure features/val contains .npy")

y_true = []
y_pred = []
print(f"[INFO] Evaluating {len(files)} files from {FEATURE_DIR_VAL} ...")
for f in files:
    label = int(os.path.basename(f).split("__")[-1].replace(".npy",""))
    x = load_and_prepare(f)
    p = float(model.predict(x[np.newaxis,...], verbose=0)[0,0])
    y_true.append(label)
    y_pred.append(p)

y_pred_label = [1 if p>=0.5 else 0 for p in y_pred]

print("N val:", len(y_true))
print("AUC:", roc_auc_score(y_true, y_pred) if len(set(y_true))>1 else "only one class in val")
print("Accuracy:", accuracy_score(y_true, y_pred_label))
print("Precision:", precision_score(y_true, y_pred_label, zero_division=0))
print("Recall:", recall_score(y_true, y_pred_label, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred_label))
print("\nPer-file predictions:")
for f, p, lab in zip(files, y_pred, y_true):
    print(f"{os.path.basename(f):40s}  pred={p:.3f}  label={lab}")
