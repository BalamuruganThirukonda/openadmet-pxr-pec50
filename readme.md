# 🧪 PXR ADMET pEC50 Prediction Challenge

This repository contains a machine learning pipeline for predicting **pEC50 values** of small molecules targeting the **Pregnane X Receptor (PXR)** using chemical structure (SMILES) and assay-derived features.

The solution combines:
- RDKit molecular descriptors
- Multiple chemical fingerprints
- Assay aggregation features
- Gradient boosting models (LightGBM)
- Cross-validation + ensemble prediction strategy

---

## 📌 Problem Statement

Given a molecule represented as a **SMILES string**, predict its **pEC50 (potency value)** for PXR activity.

The dataset includes:
- Train set with known pEC50 values
- Test set for prediction submission
- Auxiliary assay data:
  - Single concentration assay
  - Counter assay measurements

---

## 🧠 Approach Overview

We build a hybrid model combining:

### 1. Chemical Representation
- Morgan Fingerprints (ECFP-like)
- RDKit path fingerprints
- MACCS keys
- Physicochemical descriptors (MW, LogP, TPSA, etc.)
- Structural features (rings, rotatable bonds, heteroatoms)
- E-state indices
- Functional group flags (e.g., nitro group)

### 2. Assay-Derived Features
- Single concentration response aggregation
- Counter assay potency (pEC50, Emax, baseline response)
- Dose-response statistics (mean, std, monotonicity, range)
- Missingness indicators

### 3. Feature Engineering
Key engineered signals:
- Dose-response ratios
- Effect strength
- Assay consistency metrics
- Confidence scores (t-statistic × FDR signal)
- Potency × assay interaction terms

---

## 🏗 Model Architecture

Two LightGBM regression models:

### Model A — Chemistry Only
Uses only SMILES-derived molecular features.

### Model B — Chemistry + Assay Features
Adds biological assay-derived signals.

### Final Prediction Strategy
if assay data available:
use Model B
else:
use Model A


---

## 🔁 Training Strategy

- 5-Fold Cross Validation (KFold, shuffled)
- LightGBM Regressors:
  - Model A: moderate complexity
  - Model B: higher capacity + regularization
- Ensemble evaluation via OOF predictions
- Ablation study comparing:
  - Model A
  - Model B
  - Weighted ensemble

---

## 📊 Evaluation Metrics

Primary metric:
- **MAE (Mean Absolute Error)**

Secondary metric:
- **R² score**

Additional analyses:
- Error distribution by assay availability
- Worst-case prediction analysis
- Feature importance ranking

---

## 🧬 Feature Engineering Summary

### Fingerprints
- Morgan (2048-bit)
- RDKit (2048-bit)
- MACCS keys (167-bit)

### Descriptors
- RDKit physicochemical properties
- Topological indices
- Lipinski features
- QED, TPSA, LogP, MR

### Structural Features
- Rings, heterocycles, spiro atoms
- Aromaticity & flexibility
- Fraction sp³ carbon

### Assay Features
- pEC50 aggregation
- Emax (signal vs baseline)
- Dose-response curves (1µM → 99µM)

---

## 📦 Output Format

Final submission file:

| Column | Description |
|--------|------------|
| SMILES | Molecule structure |
| Molecule Name | Identifier |
| pEC50 | Predicted potency |

Generated file:
final_submission.csv




---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn lightgbm rdkit-pypi datasets matplotlib
```

## 📓 Notebook Usage

All steps of the pipeline — including:
- data loading
- feature engineering
- model training (LightGBM)
- cross-validation
- test inference
- submission generation

are implemented in a single Jupyter notebook:
PXR_ADMET_Model.ipynb



To reproduce results:

1. Open the notebook
2. Run all cells sequentially
3. The final submission file will be generated as:



---
## Project Structure
openadmet-pxr-pec50/
│
├── notebooks/
│   └── PXR_ADMET_Model.ipynb
│
├── outputs/
│   ├── kfold_metrics.xlsx
│   ├── feature_importance.xlsx
│   ├── ablation_results.xlsx
│   └── final_submission.csv
│
├── README.md
├── requirements.txt
└── .gitignore
