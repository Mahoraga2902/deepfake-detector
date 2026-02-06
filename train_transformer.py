# train_transformer.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import argparse

# ---------- Transformer Components ----------

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, dtype=tf.int32)
        pos_encoding = self.pos_emb(positions)
        return x + pos_encoding


# ---------- Build Model ----------

def build_transformer(sequence_length, feature_dim, embed_dim, num_heads, ff_dim, num_blocks):
    inputs = layers.Input(shape=(sequence_length, feature_dim))

    # project CNN features to embed dim
    x = layers.Dense(embed_dim, activation="relu")(inputs)

    # add positional embedding
    x = PositionalEmbedding(max_len=sequence_length, embed_dim=embed_dim)(x)

    # transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    return model


# ---------- Dataset Helpers ----------

def get_label(filename):
    label = filename.split("__")[-1].replace(".npy", "")
    return int(label)

def load_feature(path, seq_len):
    arr = np.load(path)
    T, D = arr.shape

    if T >= seq_len:
        return arr[:seq_len]
    else:
        pad = seq_len - T
        return np.pad(arr, ((0,pad),(0,0)), constant_values=0)

def load_dataset(folder, seq_len, batch_size):
    files = sorted(glob(os.path.join(folder, "*.npy")))
    X, y = [], []

    for f in files:
        X.append(load_feature(f, seq_len))
        y.append(get_label(f))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, X.shape[-1]   # returns dataset + feature dimension


# ---------- Main Training Script ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--out", default="deepfake_transformer.h5")
    args = parser.parse_args()

    train_folder = os.path.join(args.feature_dir, "train")
    val_folder = os.path.join(args.feature_dir, "val")

    train_ds, feature_dim = load_dataset(train_folder, args.seq_len, args.batch)
    val_ds, _ = load_dataset(val_folder, args.seq_len, args.batch)

    model = build_transformer(
        sequence_length=args.seq_len,
        feature_dim=feature_dim,
        embed_dim=args.embed,
        num_heads=args.heads,
        ff_dim=args.ff,
        num_blocks=args.blocks,
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_auc", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=6, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print(f"\nTraining finished. Best model saved to: {args.out}")


if __name__ == "__main__":
    main()
