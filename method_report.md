# 📄 OpenADMET PXR Challenge – Method Report

---

## 1. Overview

This work presents a hybrid machine learning pipeline for predicting **pEC50 values** in the OpenADMET PXR activity prediction challenge. The approach combines:

- Multi-scale molecular representations (fingerprints + descriptors)  
- Assay-derived biological response features  
- Gradient boosting regression (LightGBM)  
- Ensemble learning across complementary feature spaces  

The final submission is generated using predictions over all **513 test compounds**, as required by the competition.

---

## 2. Data Sources

The following datasets from the OpenADMET PXR Challenge were used:

- Training set (`train`)  
- Test set (`test`, 513 compounds total)  
- Counter assay dataset (`counter_assay`)  
- Single concentration assay dataset (`single_concentration`)  
- Phase 1 unblinded dataset (`phase_1_unblinded`)  

---

## 2.1 Phase 1 Unblinded Data Usage

After completion of Phase 1, the unblinded dataset was incorporated into training:

- Overlap with original training data was identified and removed to avoid duplication  
- Remaining Phase 1 compounds were appended to the training set  
- No Phase 2 test labels were used at any stage of model development  

This ensures no data leakage from the evaluation set.

---

## 3. Feature Engineering

A multi-view feature representation was constructed, combining chemical structure, physicochemical properties, and assay-derived signals.

---

### 3.1 Molecular Fingerprints

Three complementary fingerprint types were used:

- Morgan fingerprints (ECFP4, radius=2, 2048-bit)  
- RDKit path fingerprints (2048-bit)  
- MACCS structural keys (167-bit)  

These capture local substructures, paths, and predefined chemical alerts.

---

### 3.2 Physicochemical Descriptors

Computed using RDKit:

- Molecular weight, logP, molar refractivity  
- H-bond donors/acceptors  
- Rotatable bonds  
- Fraction CSP3  
- Aromaticity metrics  

---

### 3.3 Structural & Topological Features

Structural complexity and topology were encoded using:

- Ring counts (aromatic, aliphatic, heterocycles)  
- Heteroatom counts  
- Amide and spiro atom counts  
- Graph-based descriptors (BertzCT, BalabanJ, Kappa indices)  
- TPSA and heavy atom counts  

---

### 3.4 Ratio-Based Features

To capture normalized chemical effects:

- logP / molecular weight  
- TPSA / molecular weight  
- heteroatom density  
- heavy atom ratios  
- TPSA-to-logP interaction ratio  

---

### 3.5 E-state Features

E-state indices were used to capture electronic environment:

- mean, std, max, min of E-state indices  
- count-based descriptors  

---

### 3.6 Assay-Derived Features

Biological response signals were integrated from multiple experimental sources:

**Single-concentration assay:**
- log2 fold-change at 1µM, 8µM, 33µM, 99µM  
- aggregated statistics (mean, max, min, std)  
- t-statistics and FDR-adjusted significance  

**Counter assay:**
- pEC50 (mean, max)  
- Emax estimates (baseline and positive control normalized)  
- assay response differentials  

---

### 3.7 Dose–Response Engineering

Derived features include:

- Dose ratio (99µM / 1µM)  
- Dose difference and range  
- Mean and standard deviation across dose points  
- Dose coefficient of variation  
- Monotonicity score across concentrations  
- Log-transformed response strength features  
- Assay consistency metrics  

---

## 4. Feature Matrix Construction

Two feature sets were created:

### Model A (Chemistry-only)

Includes:

- Fingerprints  
- RDKit descriptors  
- Structural/topological features  
- Ratio-based descriptors  
- E-state features  
- Functional group flags  

---

### Model B (Hybrid model)

Includes:

- Model A features  
- Assay-derived biological features (single + counter assay signals)  

---

## 5. Model Architecture

Two LightGBM regressors were trained:

### Model A

- Gradient boosting model using only chemical structure features  
- Captures intrinsic structure–activity relationships  

### Model B

- Gradient boosting model using chemical + assay features  
- Captures both structure and experimental response signals  

---

## 6. Training Strategy

- 5-fold cross-validation (KFold, shuffled, seed=42)  
- Out-of-fold predictions generated for both models  
- Metrics:
  - Mean Absolute Error (MAE)  
  - R² score  

Final models were retrained on the full dataset after validation.

---

## 7. Ensemble Strategy

Final prediction is a routing-based ensemble:

- If assay data is available → use Model B  
- Otherwise → use Model A  

This ensures optimal use of experimental data when present.

---

## 8. Inference Pipeline

For test inference:

- All 513 compounds in the official test set were used  
- Identical feature engineering pipeline applied to test data  
- Predictions generated using both models  
- Final output selected via assay-aware routing logic  

---

## 9. Submission Construction

The final submission file:

- Contains all 513 test compounds  
- Preserves original SMILES and Molecule Name identifiers  
- Attaches predicted pEC50 values  
- Ensures full alignment with competition scoring format  

---

## 10. Software Stack

- Python  
- RDKit (molecular descriptors & fingerprints)  
- LightGBM (gradient boosting regression)  
- scikit-learn (validation & metrics)  
- HuggingFace Datasets (data loading)  
- NumPy / Pandas  

---

## 11. Reproducibility

The pipeline is fully reproducible with:

- Fixed random seed (42)  
- Deterministic feature extraction  
- Consistent preprocessing between train and test  
- No use of Phase 2 labels during training  

---

## 12. Key Design Choices

- Multi-representation molecular encoding improves generalization  
- Assay integration provides biological context beyond structure  
- Routing-based ensemble improves robustness for missing assay data  
- Phase 1 unblinded data increases training diversity without leakage  

---

## 13. Ethical / Data Integrity Statement

All external data usage strictly follows competition rules. Phase 1 unblinded data was used only after release and was not used for validation leakage into Phase 2 test predictions.
