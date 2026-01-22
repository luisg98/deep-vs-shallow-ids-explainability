import os
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import json

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_PATH = "data/processed/CICIDS2017"
SAVE_DIR = "Model/CICIDS2017/MLP"
RESULTS_DIR = "Results/CICIDS2017/MLP/Model"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
print("\nLoading dataset...")
df = dd.read_parquet(DATA_PATH)
pdf = df.compute()
pdf.columns = pdf.columns.str.strip()
print("Loaded:", pdf.shape)

# Separate features / labels
y = pdf["Label"]
X = pdf.drop(columns=["Label"])

# Numeric conversion + cleaning
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean()).fillna(0)

# Binary target
y_binary = (y != "BENIGN").astype(int)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.30, stratify=y_binary, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]

# Build MLP
mlp = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(num_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining MLP...")
early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

start = time.time()
history = mlp.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early],
    verbose=1
)
train_time = time.time() - start
print(f"Training time: {train_time:.2f}s")

# Best threshold (using VALIDATION)
val_scores = mlp.predict(X_val).ravel()

best_thr, best_f1 = 0.5, 0
for thr in np.linspace(0.1, 0.9, 81):
    preds = (val_scores > thr).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1

print(f"\nBest threshold = {best_thr:.3f} (Val F1={best_f1:.4f})")

# Test Evaluation
test_scores = mlp.predict(X_test).ravel()
test_preds = (test_scores > best_thr).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# Save model + artefacts
mlp.save(f"{SAVE_DIR}/mlp_model.h5")
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{SAVE_DIR}/feature_names.pkl")

with open(f"{SAVE_DIR}/best_threshold.txt", "w") as f:
    f.write(str(best_thr))

with open(f"{SAVE_DIR}/training_history.json", "w") as f:
    json.dump(history.history, f, indent=4)

print("\nSaved MLP model and artefacts successfully!")

# GRÁFICOS DO TREINO

# ---------- Learning curve (loss) ----------
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("MLP - Learning Curve (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(f"{RESULTS_DIR}/learning_curve_loss.png")
plt.close()

# ---------- Accuracy curve ----------
plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("MLP - Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
plt.close()

# ---------- ROC curve ----------
fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("MLP - ROC Curve")
plt.legend()
plt.grid()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png")
plt.close()

# ---------- PR curve ----------
precision, recall, _ = precision_recall_curve(y_test, test_scores)

plt.figure(figsize=(7,6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("MLP - Precision-Recall Curve")
plt.grid()
plt.savefig(f"{RESULTS_DIR}/pr_curve.png")
plt.close()

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("MLP - Confusion Matrix")
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()

# ---------- F1 vs threshold ----------
thresholds = np.linspace(0.1, 0.9, 200)
f1_scores = [f1_score(y_val, (val_scores > thr)) for thr in thresholds]

plt.figure(figsize=(7,5))
plt.plot(thresholds, f1_scores)
plt.axvline(best_thr, color="red", linestyle="--")
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.title("MLP - F1 vs Threshold")
plt.grid()
plt.savefig(f"{RESULTS_DIR}/f1_vs_threshold.png")
plt.close()

# ---------- Score distribution ----------
plt.figure(figsize=(7,5))
plt.hist(test_scores, bins=50, alpha=0.7)
plt.axvline(best_thr, color="red", linestyle="--")
plt.title("MLP - Score Distribution")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.grid()
plt.savefig(f"{RESULTS_DIR}/score_distribution.png")
plt.close()

print("\nGenerated ALL MLP training plots!\n")
