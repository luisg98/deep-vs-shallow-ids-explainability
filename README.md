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

