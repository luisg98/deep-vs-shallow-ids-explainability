# Deep vs Shallow IDS Explainability

Experimental pipeline comparing MLP (shallow) and CNN (deep) architectures for binary intrusion detection, with SHAP-based explanation stability analysis.

Evaluated on CICIDS2017 and CICTONIOT. Benchmarks go beyond accuracy: F1, ROC, PR curves, and explanation stability across samples.

## Run

```bash
python 31_ModelExplainingCICIDS2017MLP.py
python 32_ModelExplainingCICIDS2017CNN.py
```

Datasets in Parquet format, numerical features only, with a `Label` column.

## Stack

Python, PyTorch, scikit-learn, SHAP.
