# ============================================================  
# IMPORT LIBRARIES
# ============================================================
from datasets import load_dataset

import os
import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, QED
from rdkit.Chem import EState
from rdkit.Chem import MACCSkeys

from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

os.makedirs("C:\\Users\\User\\OneDrive - ingenium digital diagnostics GmbH\\Desktop\\Datascience competition\\PxR_prediction\\outputs", exist_ok=True)
# ============================================================
# LOAD DATASET
# ============================================================  
ds = load_dataset("OpenADMET/PXR-Challenge-Train-Test")

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

counter = load_dataset(
    "OpenADMET/PXR-Challenge-Train-Test",
    "counter_assay"
)["train"].to_pandas()

unblind = load_dataset(
    "OpenADMET/PXR-Challenge-Train-Test",
    "phase_1_unblinded"
)["test"].to_pandas()

single = load_dataset(
    "OpenADMET/PXR-Challenge-Train-Test",
    "single_concentration"
)["train"].to_pandas()

print("Initial shapes:")
print(train_df.shape, test_df.shape, unblind.shape, single.shape)
# =====================================================
# 3. LEAKAGE-SAFE DATA MERGE (IMPORTANT)
# =====================================================

# 3.1 Find overlap
overlap = set(train_df["Molecule Name"]).intersection(set(unblind["Molecule Name"]))
print("Overlap before cleaning:", len(overlap))

# 3.2 REMOVE overlap from unblind (critical safety step)
unblind_clean = unblind[~unblind["Molecule Name"].isin(train_df["Molecule Name"])].copy()

# 3.3 Add unblinded to training
train_df = pd.concat([train_df, unblind_clean], ignore_index=True)


print("Final shapes:")
print("Train:", train_df.shape)
print("Test :", test_df.shape)
# =====================================================
# 4. PREP SINGLE + COUNTER ASSAYS
# =====================================================
single = single[[
    "SMILES","concentration_M","log2_fc_estimate","t_statistic","neg_log10_fdr"
]]

def conc_label(x):
    if x < 2e-6: return "1uM"
    elif x < 2e-5: return "8uM"
    elif x < 6e-5: return "33uM"
    else: return "99uM"

single["conc_label"] = single["concentration_M"].apply(conc_label)

single_pivot = single.pivot_table(
    index="SMILES",
    columns="conc_label",
    values="log2_fc_estimate",
    aggfunc="mean"
).reset_index()

single_agg = single.groupby("SMILES").agg({
    "log2_fc_estimate": ["mean","max","min","std"],
    "t_statistic":"mean",
    "neg_log10_fdr":"mean"
})

single_agg.columns = ["_".join(c) for c in single_agg.columns]
single_agg = single_agg.reset_index()


counter = counter[[
    "SMILES",
    "pEC50",
    "Emax_estimate (log2FC vs. baseline)",
    "Emax.vs.pos.ctrl_estimate (dimensionless)"
]]

counter = counter.rename(columns={
    "pEC50":"counter_pEC50",
    "Emax_estimate (log2FC vs. baseline)":"counter_emax",
    "Emax.vs.pos.ctrl_estimate (dimensionless)":"counter_emax_ctrl"
})

counter_agg = counter.groupby("SMILES").agg({
    "counter_pEC50":["mean","max"],
    "counter_emax":["mean","max"],
    "counter_emax_ctrl":["mean","max"]
})

counter_agg.columns = ["_".join(c) for c in counter_agg.columns]
counter_agg = counter_agg.reset_index()
# =====================================================
# 5. BUILD TRAIN DF
# =====================================================
df = train_df.merge(counter_agg, on="SMILES", how="left")
df = df.merge(single_pivot, on="SMILES", how="left")
df = df.merge(single_agg, on="SMILES", how="left")
# =====================================================
# 6. ASSAY FEATURES
# =====================================================
ASSAY_COLS = ["1uM","8uM","33uM","99uM","counter_pEC50_mean"]

for c in ASSAY_COLS:
    df[c + "_miss"] = df.get(c, np.nan).isna().astype(int)

df["has_assay"] = df[ASSAY_COLS].notna().any(axis=1).astype(int)

df[ASSAY_COLS] = df[ASSAY_COLS].fillna(-999)
# =====================================================
# 7. FEATURE ENGINEERING (same as Phase 1)
# =====================================================
DOSE_COLS = ["1uM","8uM","33uM","99uM"]

df["dose_ratio"] = df["99uM"] / (df["1uM"] + 1e-6)
df["dose_diff"] = df["99uM"] - df["1uM"]

df["dose_mean"] = df[DOSE_COLS].mean(axis=1)
df["dose_std"] = df[DOSE_COLS].std(axis=1)

df["effect_strength"] = df["log2_fc_estimate_max"] - df["log2_fc_estimate_min"]
df["confidence"] = df["neg_log10_fdr_mean"] * df["t_statistic_mean"]

df["emax_gap"] = df["counter_emax_mean"] - df["counter_emax_ctrl_mean"]

df["dose_ratio_log"] = np.log1p(np.abs(df["dose_ratio"]))
df["effect_strength_log"] = np.log1p(np.abs(df["effect_strength"]))
# =====================================================
# Remove CI columns (if any) to avoid leakage
# =====================================================
df = df.drop(columns=[c for c in df.columns if "ci" in c.lower()], errors="ignore")
# =====================================================
# Tabular features
# ==================================================== 
X_tab = df.select_dtypes(include=[np.number]).drop(columns=["pEC50"])
X_tab = X_tab.loc[:, X_tab.nunique() > 1]

TAB_FEATURES = X_tab.columns.tolist()
y = df["pEC50"].values
# ====================================================
# SMILES features   
# =====================================================

# =====================================================
# 0. FINGERPRINT GENERATORS
# =====================================================
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
rdkit_gen = GetRDKitFPGenerator(fpSize=2048)


# =====================================================
# 1. MORGAN FINGERPRINT (ECFP4 STYLE)
# =====================================================
def smiles_to_morgan(s):
    mol = Chem.MolFromSmiles(s)
    return np.array(morgan_gen.GetFingerprint(mol)) if mol else np.zeros(2048)


# =====================================================
# 2. RDKit PATH FINGERPRINT (IMPORTANT DIVERSITY BOOST)
# =====================================================
def smiles_to_rdkitfp(s):
    mol = Chem.MolFromSmiles(s)
    return np.array(rdkit_gen.GetFingerprint(mol)) if mol else np.zeros(2048)


# =====================================================
# 3. MACCS KEYS (HIGH INFORMATION CHEMICAL ALERTS)
# =====================================================
def smiles_to_maccs(s):
    mol = Chem.MolFromSmiles(s)
    return np.array(MACCSkeys.GenMACCSKeys(mol)) if mol else np.zeros(167)


# =====================================================
# 4. RDKit CORE DESCRIPTORS (NON-REDUNDANT SET)
# =====================================================
CORE_DESC = [
    "MolWt","MolLogP","MolMR",
    "NumHDonors","NumHAcceptors",
    "NumRotatableBonds","FractionCSP3",
    "HeavyAtomCount","NumAromaticRings"
]

def smiles_to_desc(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*len(CORE_DESC)
    return [getattr(Descriptors, d)(mol) for d in CORE_DESC]


# =====================================================
# 5. STRUCTURAL FEATURES (DE-REDUNDANT)
# =====================================================
def smiles_to_frag(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*10
    return [
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol)
    ]


# =====================================================
# 6. DERIVED RATIO FEATURES (LOW REDUNDANCY)
# =====================================================
def smiles_to_ratio(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*5

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    return [
        logp/(mw+1e-6),
        tpsa/(mw+1e-6),
        tpsa/(abs(logp)+1e-6),
        rdMolDescriptors.CalcNumHeteroatoms(mol)/(mw+1e-6),
        rdMolDescriptors.CalcNumHeavyAtoms(mol)/(mw+1e-6)
    ]


# =====================================================
# 7. HIGH-INFORMATION GLOBAL + BOOSTING DESCRIPTORS
# =====================================================
def smiles_to_extra(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*12

    return [
        QED.qed(mol),
        Crippen.MolLogP(mol),
        Crippen.MolMR(mol),

        Descriptors.BertzCT(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.Ipc(mol),

        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),

        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),

        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol)
    ]


# =====================================================
# 8. E-STATE FEATURES (STRONG BOOST FOR ADMET)
# =====================================================
def smiles_to_estate(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*5

    e = np.array(EState.EStateIndices(mol))
    return [e.mean(), e.std(), e.max(), e.min(), len(e)]


# =====================================================
# 9. TOPOLOGICAL / GRAPH FEATURES (VERY USEFUL)
# =====================================================
def smiles_to_topo(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return [0]*5

    return [
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.BertzCT(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol)
    ]


# =====================================================
# 10. SIMPLE FUNCTIONAL GROUP FLAG
# =====================================================
def has_nitro(s):
    mol = Chem.MolFromSmiles(s)
    if not mol: return 0
    return int(mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")))
# =====================================================
# Feature Matrices
# =====================================================
X_morgan = np.array([smiles_to_morgan(s) for s in df["SMILES"]])
X_rdkit  = np.array([smiles_to_rdkitfp(s) for s in df["SMILES"]])
X_maccs  = np.array([smiles_to_maccs(s) for s in df["SMILES"]])

X_desc   = np.array([smiles_to_desc(s) for s in df["SMILES"]])
X_frag   = np.array([smiles_to_frag(s) for s in df["SMILES"]])
X_ratio  = np.array([smiles_to_ratio(s) for s in df["SMILES"]])
X_extra  = np.array([smiles_to_extra(s) for s in df["SMILES"]])
X_estate = np.array([smiles_to_estate(s) for s in df["SMILES"]])
X_topo   = np.array([smiles_to_topo(s) for s in df["SMILES"]])
X_nitro  = np.array([has_nitro(s) for s in df["SMILES"]]).reshape(-1,1)
# =====================================================
# Model Inputs
# =====================================================
# Model A = chemistry only
X_A = np.hstack([
    X_morgan,
    X_rdkit,
    X_maccs,
    X_desc,
    X_frag,
    X_ratio,
    X_extra,
    X_estate,
    X_topo,
    X_nitro
])

# Model B = chemistry + assay features
X_B = np.hstack([X_A, X_tab.values])

y = df["pEC50"].values

print("X_A:", X_A.shape)
print("X_B:", X_B.shape)

# =====================================================
# K-FOLD CROSS-VALIDATION
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_A = np.zeros(len(df))
oof_B = np.zeros(len(df))

results = []

models_A = []
models_B = []

for fold, (tr, va) in enumerate(kf.split(X_A), 1):

    print(f"\n========== FOLD {fold} ==========")

    # =====================================================
    # MODEL A
    # =====================================================

    model_A = LGBMRegressor(
        n_estimators=1500, 
        learning_rate=0.01, 
        num_leaves=64, 
        min_child_samples=20, 
        feature_fraction=0.7, 
        bagging_fraction=0.8, 
        bagging_freq=1, 
        random_state=42, 
        verbose=-1
    )

    model_A.fit(X_A[tr], y[tr])

    pred_A = model_A.predict(X_A[va])

    oof_A[va] = pred_A

    mae_A = mean_absolute_error(y[va], pred_A)
    r2_A = r2_score(y[va], pred_A)

    # =====================================================
    # MODEL B
    # =====================================================

    model_B = LGBMRegressor(
        n_estimators=2000, 
        learning_rate=0.005, 
        num_leaves=128, 
        min_child_samples=20, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        feature_fraction=0.7, 
        bagging_fraction=0.8, 
        bagging_freq=1, 
        reg_alpha=1.0, 
        reg_lambda=1.0, 
        random_state=42, 
        verbose=-1
    )

    model_B.fit(X_B[tr], y[tr])

    pred_B = model_B.predict(X_B[va])

    oof_B[va] = pred_B

    mae_B = mean_absolute_error(y[va], pred_B)
    r2_B = r2_score(y[va], pred_B)

    models_A.append(model_A)
    models_B.append(model_B)

    results.append({
        "Fold": fold,
        "MAE_Model_A": mae_A,
        "R2_Model_A": r2_A,
        "MAE_Model_B": mae_B,
        "R2_Model_B": r2_B
    })

    print(f"Model A MAE: {mae_A:.4f}")
    print(f"Model A R2 : {r2_A:.4f}")

    print(f"Model B MAE: {mae_B:.4f}")
    print(f"Model B R2 : {r2_B:.4f}")
# =====================================
# ABLATION TESTS
# =====================================

# -------------------------
# MODEL A ONLY
# -------------------------
mae_A_final = mean_absolute_error(y, oof_A)
r2_A_final = r2_score(y, oof_A)

# -------------------------
# MODEL B ONLY
# -------------------------
mae_B_final = mean_absolute_error(y, oof_B)
r2_B_final = r2_score(y, oof_B)

# -------------------------
# ENSEMBLE
# -------------------------
ensemble_pred = 0.3 * oof_A + 0.7 * oof_B

mae_ens = mean_absolute_error(y, ensemble_pred)
r2_ens = r2_score(y, ensemble_pred)

# -------------------------
# RESULTS TABLE
# -------------------------
ablation_df = pd.DataFrame({
    "Model": [
        "Model A Only",
        "Model B Only",
        "Ensemble"
    ],
    "MAE": [
        mae_A_final,
        mae_B_final,
        mae_ens
    ],
    "R2": [
        r2_A_final,
        r2_B_final,
        r2_ens
    ]
})

print("\n========================")
print("ABLATION TEST RESULTS")
print("========================")

print(ablation_df)

# Save
ablation_df.to_excel(
    "C:\\Users\\User\\OneDrive - ingenium digital diagnostics GmbH\\Desktop\\Datascience competition\\PxR_prediction\\outputs\\ablation_results_phase2.xlsx",
    index=False
)

print("\nAblation results saved → ablation_results_phase2.xlsx")
# ====================================================
# SAVE K-FOLD METRICS
# ====================================================
results_df = pd.DataFrame(results)

mean_row = {
    "Fold": "MEAN",
    "MAE_Model_A": results_df["MAE_Model_A"].mean(),
    "R2_Model_A": results_df["R2_Model_A"].mean(),
    "MAE_Model_B": results_df["MAE_Model_B"].mean(),
    "R2_Model_B": results_df["R2_Model_B"].mean()
}

results_df = pd.concat([
    results_df,
    pd.DataFrame([mean_row])
], ignore_index=True)

results_df.to_excel(
    "C:\\Users\\User\\OneDrive - ingenium digital diagnostics GmbH\\Desktop\\Datascience competition\\PxR_prediction\\outputs\\kfold_metrics_phase2.xlsx",
    index=False
)

print("Metrics saved.")
# ====================================================
# PLOT ENSEMBLE REGRESSION
# ====================================================
final_oof_A = oof_A.copy()

final_mae_A = mean_absolute_error(y, final_oof_A)
final_r2_A = r2_score(y, final_oof_A)

plt.figure(figsize=(7, 7))

plt.scatter(
    y,
    final_oof_A,
    alpha=0.5
)

# Perfect prediction line
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2
)

# Metrics text
plt.text(
    0.05,
    0.95,
    f"MAE = {final_mae_A:.4f}\nR² = {final_r2_A:.4f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.8
    )
)

plt.xlabel("True pEC50")
plt.ylabel("Predicted pEC50")
plt.title("Ensemble Regression Plot")

plt.show()
final_oof_B = oof_B.copy()

final_mae_B = mean_absolute_error(y, final_oof_B)
final_r2_B = r2_score(y, final_oof_B)

plt.figure(figsize=(7, 7))

plt.scatter(
    y,
    final_oof_B,
    alpha=0.5
)

# Perfect prediction line
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2
)

# Metrics text
plt.text(
    0.05,
    0.95,
    f"MAE = {final_mae_B:.4f}\nR² = {final_r2_B:.4f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.8
    )
)

plt.xlabel("True pEC50")
plt.ylabel("Predicted pEC50")
plt.title("Ensemble Regression Plot")

plt.show()
# ====================================================
# FINAL MODEL TRAINING
# ====================================================
final_model_A = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=64,
    min_child_samples=20,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=42,
    verbose=-1
)

final_model_B = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.005,
    num_leaves=128,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

final_model_A.fit(X_A, y)
final_model_B.fit(X_B, y)
# ============================================================
# 20. BUILD TEST FEATURES (MATCH TRAIN EXACTLY)
# ============================================================

test = test_df.copy()

# -------------------------
# MERGE ASSAY DATA
# -------------------------
test = test.merge(
    counter_agg,
    on="SMILES",
    how="left"
)

test = test.merge(
    single_pivot,
    on="SMILES",
    how="left"
)

test = test.merge(
    single_agg,
    on="SMILES",
    how="left"
)

# ============================================================
# ASSAY FEATURES (IDENTICAL TO TRAIN)
# ============================================================

ASSAY_COLS = [
    "1uM",
    "8uM",
    "33uM",
    "99uM",
    "counter_pEC50_mean"
]

for c in ASSAY_COLS:
    if c not in test.columns:
        test[c] = np.nan

for c in ASSAY_COLS:
    test[c + "_miss"] = (
        test[c]
        .isna()
        .astype(int)
    )

test["has_assay"] = (
    test[ASSAY_COLS]
    .notna()
    .any(axis=1)
    .astype(int)
)

# EXACT SAME IMPUTATION
test[ASSAY_COLS] = (
    test[ASSAY_COLS]
    .fillna(-999)
)

# ============================================================
# FEATURE ENGINEERING (MATCH TRAIN EXACTLY)
# ============================================================

DOSE_COLS = [
    "1uM",
    "8uM",
    "33uM",
    "99uM"
]

# -------------------------
# Basic dose features
# -------------------------
test["dose_ratio"] = (
    test["99uM"] /
    (test["1uM"] + 1e-6)
)

test["dose_diff"] = (
    test["99uM"] -
    test["1uM"]
)

test["dose_mean"] = (
    test[DOSE_COLS]
    .mean(axis=1)
)

test["dose_std"] = (
    test[DOSE_COLS]
    .std(axis=1)
)

# -------------------------
# Assay signal features
# -------------------------
test["effect_strength"] = (
    test["log2_fc_estimate_max"] -
    test["log2_fc_estimate_min"]
)

test["confidence"] = (
    test["neg_log10_fdr_mean"] *
    test["t_statistic_mean"]
)

test["emax_gap"] = (
    test["counter_emax_mean"] -
    test["counter_emax_ctrl_mean"]
)

# -------------------------
# Log transforms
# -------------------------
test["dose_ratio_log"] = np.log1p(
    np.abs(test["dose_ratio"])
)

test["effect_strength_log"] = np.log1p(
    np.abs(test["effect_strength"])
)

# ============================================================
# EXTRA ENGINEERED FEATURES
# (YOU WERE MISSING MOST OF THESE)
# ============================================================

test["potency_x_confidence"] = (
    test["counter_pEC50_mean"] *
    test["confidence"]
)

test["effect_x_max"] = (
    test["effect_strength"] *
    test["counter_emax_mean"]
)

test["dose_response_strenght"] = (
    test["dose_mean"] *
    test["effect_strength"]
)

test["assay_consistency"] = (
    test["dose_std"] /
    (test["dose_mean"] + 1e-6)
)

test["emax_ratio"] = (
    test["counter_emax_mean"] /
    (
        np.abs(
            test["counter_emax_ctrl_mean"]
        ) + 1e-6
    )
)

test["dose_max"] = (
    test[DOSE_COLS]
    .max(axis=1)
)

test["dose_min"] = (
    test[DOSE_COLS]
    .min(axis=1)
)

test["dose_range"] = (
    test["dose_max"] -
    test["dose_min"]
)

test["dose_cv"] = (
    test["dose_std"] /
    (
        np.abs(test["dose_mean"])
        + 1e-6
    )
)

test["high_low_ratio"] = (
    test["99uM"] /
    (
        np.abs(test["1uM"])
        + 1e-6
    )
)

test["dose_monotonicity"] = (
    (test["99uM"] > test["33uM"]).astype(int)
    +
    (test["33uM"] > test["8uM"]).astype(int)
    +
    (test["8uM"] > test["1uM"]).astype(int)
)

# ============================================================
# CLEANUP
# ============================================================

test = test.drop(
    columns=[
        c for c in test.columns
        if "ci" in c.lower()
    ],
    errors="ignore"
)

# ============================================================
# GUARANTEE SAME FEATURE ORDER AS TRAIN
# ============================================================

X_test_tab = (
    test
    .reindex(
        columns=TAB_FEATURES,
        fill_value=0
    )
    .fillna(0)
    .values
)

print("Test tabular shape:", X_test_tab.shape)
#============================================================
# 21. BUILD TEST CHEMISTRY FEATURES
#============================================================
print("\nBuilding test chemistry features...")

X_test_morgan = np.array([
    smiles_to_morgan(s)
    for s in test["SMILES"]
])

X_test_rdkit = np.array([
    smiles_to_rdkitfp(s)
    for s in test["SMILES"]
])

X_test_maccs = np.array([
    smiles_to_maccs(s)
    for s in test["SMILES"]
])

X_test_desc = np.array([
    smiles_to_desc(s)
    for s in test["SMILES"]
])

X_test_frag = np.array([
    smiles_to_frag(s)
    for s in test["SMILES"]
])

X_test_ratio = np.array([
    smiles_to_ratio(s)
    for s in test["SMILES"]
])

X_test_extra = np.array([
    smiles_to_extra(s)
    for s in test["SMILES"]
])

X_test_estate = np.array([
    smiles_to_estate(s)
    for s in test["SMILES"]
])

X_test_topo = np.array([
    smiles_to_topo(s)
    for s in test["SMILES"]
])

X_test_nitro = np.array([
    has_nitro(s)
    for s in test["SMILES"]
]).reshape(-1, 1)

# -------------------------
# MODEL A
# -------------------------
X_test_A = np.hstack([
    X_test_morgan,
    X_test_rdkit,
    X_test_maccs,
    X_test_desc,
    X_test_frag,
    X_test_ratio,
    X_test_extra,
    X_test_estate,
    X_test_topo,
    X_test_nitro
])

# -------------------------
# MODEL B
# -------------------------
X_test_B = np.hstack([
    X_test_A,
    X_test_tab
])

print("X_test_A:", X_test_A.shape)
print("X_test_B:", X_test_B.shape)
# ============================================================
# 22. TEST PREDICTIONS
# ============================================================

pred_A = final_model_A.predict(X_test_A)
pred_B = final_model_B.predict(X_test_B)

# use assay-aware routing
has_assay = (
    test["has_assay"]
    .values
    .astype(bool)
)

final_pred = np.where(
    has_assay,
    pred_B,
    pred_A
)

print("Prediction finished.")

# ============================================================
# 23. BUILD FULL SUBMISSION SET (513 compounds)
# ============================================================

# Start from ORIGINAL test_df (DO NOT FILTER IT EARLY)
full_test = ds["test"].to_pandas().copy()

# Merge predictions back using Molecule Name
pred_df = pd.DataFrame({
    "Molecule Name": test["Molecule Name"],
    "pEC50": final_pred
})

# Join predictions onto full test set
submission = full_test[["SMILES", "Molecule Name"]].merge(
    pred_df,
    on="Molecule Name",
    how="left"
)

# Safety check
print("Missing predictions:", submission["pEC50"].isna().sum())

# Save
submission.to_csv(
    r"C:\Users\User\OneDrive - ingenium digital diagnostics GmbH\Desktop\Datascience competition\PxR_prediction\outputs\final_submission_phase2.csv",
    index=False
)

print("Saved → final_submission_phase2.csv")
print(submission.head())