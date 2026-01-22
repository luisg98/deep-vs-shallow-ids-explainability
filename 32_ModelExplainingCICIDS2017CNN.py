import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "data/processed/CICIDS2017"
MODEL_DIR = "Model/CICIDS2017/CNN"
RESULTS_DIR = "Results/CICIDS2017/CNN/SHAP"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = dd.read_parquet(DATA_PATH)
pdf = df.compute()
pdf.columns = pdf.columns.str.strip()

y = pdf["Label"]
X = pdf.drop(columns=["Label"])

# Cleaning
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean()).fillna(0)

y_binary = (y != "BENIGN").astype(int)

# Split
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.30, stratify=y_binary, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Load model + scaler
model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn_model.h5")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# Scale data
X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)

# Sampling
sample_size = 800
background_size = 50

idx = np.random.choice(len(X_test_scaled), sample_size, replace=False)
b_idx = np.random.choice(len(X_train_scaled), background_size, replace=False)

X_sample = X_test_scaled[idx]      
background = X_train_scaled[b_idx] 

print("Shapes → X_sample:", X_sample.shape, "background:", background.shape)

# FIX: WRAP MODEL.PREDICT
def fast_predict(data):
    return model.predict(data, verbose=0).reshape(-1)

# CORRECT SHAP MASKER: Partition Masker 
masker = shap.maskers.Partition(background)

# CORRECT SHAP EXPLAINER FOR CNNs
explainer = shap.explainers.Partition(fast_predict, masker)

# Compute SHAP values
shap_values = explainer(X_sample).values
print("SHAP values:", shap_values.shape)

# Save arrays
np.save(f"{RESULTS_DIR}/shap_values.npy", shap_values)
np.save(f"{RESULTS_DIR}/X_sample.npy", X_sample)
pd.DataFrame({"feature": feature_names}).to_csv(f"{RESULTS_DIR}/features.csv", index=False)

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(f"{RESULTS_DIR}/shap_importance.csv", index=False)


# BARPLOT
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"][:20][::-1],
    importance_df["importance"][:20][::-1]
)
plt.title("Top-20 SHAP Feature Importance (CNN)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/barplot_top20.png", dpi=200)
plt.close()

# SUMMARY PLOT
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.savefig(f"{RESULTS_DIR}/summary_plot.png", dpi=200)
plt.close()

# HEATMAP
plt.figure(figsize=(16, 8))
shap.plots.heatmap(
    shap.Explanation(
        values=shap_values,
        data=X_sample,
        feature_names=feature_names
    ),
    max_display=20,
    show=False
)
plt.savefig(f"{RESULTS_DIR}/heatmap.png", dpi=200)
plt.close()

# FIX expected value
expected_value = np.mean(fast_predict(background))

# DECISION PLOT
plt.figure(figsize=(16, 8))
shap.decision_plot(
    expected_value,
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.savefig(f"{RESULTS_DIR}/decision_plot.png", dpi=200)
plt.close()

# DEPENDENCE PLOTS
for feat in importance_df["feature"][:5]:

    safe_feat = str(feat)
    for bad in [" ", "/", "\\", ":", "(", ")", ";", ","]:
        safe_feat = safe_feat.replace(bad, "_")

    plt.figure(figsize=(8,6))
    shap.dependence_plot(
        feat,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/dependence_{safe_feat}.png", dpi=200)
    plt.close()


print("CNN SHAP Analysis Completed (Partition Explainer used successfully!)")
