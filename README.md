# Deep vs Shallow IDS Explainability

This repository implements a complete experimental pipeline to **compare shallow and deep learning models for Intrusion Detection Systems (IDS)**, with a strong focus on **model explainability and explanation stability**.

The project evaluates **MLP (shallow)** and **CNN (deep)** architectures on two widely used IDS datasets, combining **robust performance evaluation** with **state-of-the-art SHAP-based explainability**.

---

## 🔍 Objectives

- Compare **shallow vs deep neural networks** for binary intrusion detection
- Evaluate detection performance beyond accuracy (F1, ROC, PR)
- Apply **explainable AI (XAI)** techniques to IDS models
- Study the **stability of explanations** across different samples
- Provide a fully reproducible experimental pipeline

---

## 📊 Datasets

- **CICIDS2017**
- **CICTONIOT**

Datasets are expected to be **preprocessed and stored in Parquet format** under:

python 31_ModelExplainingCICIDS2017MLP.py
python 32_ModelExplainingCICIDS2017CNN.py


Each dataset must contain:
- Numerical features only
- A `Label` column
  - CICIDS2017: `BENIGN` vs attack
  - CICTONIOT: binary labels (0/1)

---

## 🧠 Models

### Shallow Model
- **Multilayer Perceptron (MLP)**
- Fully connected architecture
- Dropout regularization
- Binary Cross-Entropy loss

### Deep Model
- **1D Convolutional Neural Network (CNN)**
- Conv1D + BatchNorm + MaxPooling
- Designed for tabular IDS features
- Binary Cross-Entropy loss

---

## ⚙️ Training Strategy

- Train / Validation / Test split: **70% / 15% / 15%**
- Stratified sampling
- Feature standardization (`StandardScaler`)
- Early stopping based on validation loss
- **Optimal decision threshold selected using validation F1-score**

All models save:
- Trained weights
- Scaler
- Feature names
- Best threshold
- Training history
- Performance plots

---

## 📈 Evaluation Metrics

- Precision, Recall, F1-score
- ROC curve & AUC
- Precision–Recall curve
- Confusion matrix
- Learning curves
- Score distribution
- F1-score vs threshold

---

## 🔎 Explainability (SHAP)

### MLP
- **SHAP KernelExplainer**
- Global and local explanations
- Feature importance ranking
- Beeswarm, bar plots, heatmaps, decision plots

### CNN
- **SHAP PartitionExplainer** (deep-model appropriate)
- Efficient and stable explanations
- Correct handling of expected values
- Same visualization suite as MLP

All SHAP outputs are saved to disk for further analysis.

---

## 🔁 Stability Analysis

Two Jupyter notebooks perform **stability analysis of SHAP explanations**:
- Repeated sampling
- Comparison of feature rankings
- Robustness evaluation across runs

---


