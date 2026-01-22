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
MODEL_DIR = "Model/CICIDS2017/MLP"
RESULTS_DIR = "Results/CICIDS2017/MLP/SHAP"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = dd.read_parquet(DATA_PATH)
pdf = df.compute()
pdf.columns = pdf.columns.str.strip()

y = pdf["Label"]
X = pdf.drop(columns=["Label"])

# Cleaning
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean()).fillna(0)

# Binary target
y_binary = (y != "BENIGN").astype(int)

# Split
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Load model
model = tf.keras.models.load_model(f"{MODEL_DIR}/mlp_model.h5")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# Scale data
X_test_scaled = scaler.transform(X_test)

# SHAP sampling
sample_size = 500
idx = np.random.choice(len(X_test_scaled), sample_size, replace=False)
X_sample = X_test_scaled[idx]

background_size = 100
background_idx = np.random.choice(len(X_train), background_size, replace=False)
background = scaler.transform(X_train.iloc[background_idx])

print(f"Sample size for SHAP: {sample_size}")
print(f"Background size: {background_size}")

# SHAP Explainer
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_sample, nsamples=500)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 0]

shap_values = np.array(shap_values)

# Save raw SHAP values
np.save(f"{RESULTS_DIR}/shap_values.npy", shap_values)
np.save(f"{RESULTS_DIR}/X_sample.npy", X_sample)
pd.DataFrame({"feature": feature_names}).to_csv(f"{RESULTS_DIR}/features.csv", index=False)

print("Saved SHAP base files!")

# Global Feature Importance
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(f"{RESULTS_DIR}/shap_importance.csv", index=False)

#  PLOTS  
#  1. Bar Plot (TOP-20)
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"][:20][::-1], importance_df["importance"][:20][::-1])
plt.title("Top-20 SHAP Global Feature Importance (MLP)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/barplot_top20.png", dpi=200)
plt.close()

#  2. Summary Plot (Beeswarm)
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.savefig(f"{RESULTS_DIR}/summary_plot.png", dpi=200)
plt.close()

# 3. Heatmap of SHAP values
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
plt.savefig(f"{RESULTS_DIR}/shap_heatmap.png", dpi=200)
plt.close()

# 4. Decision plot (global)
plt.figure(figsize=(16, 8))
shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values,
    features=X_sample,
    feature_names=feature_names,
    show=False
)
plt.savefig(f"{RESULTS_DIR}/decision_plot.png", dpi=200)
plt.close()

# 5. Dependence plots for top 5 features
for feat in importance_df["feature"][:5]:

    plt.figure(figsize=(8,6))
    shap.dependence_plot(
        feat,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()

    safe_feat = str(feat).replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_feat = safe_feat.replace("(", "_").replace(")", "_").replace(":", "_")

    plt.savefig(f"{RESULTS_DIR}/dependence_{safe_feat}.png", dpi=200)
    plt.close()



print("Generated ALL SHAP visualizations for MLP!")
