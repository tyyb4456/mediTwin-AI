"""
Digital Twin — Risk Model Training Script
==========================================
Run this ONCE before starting the Digital Twin Agent.

Trains 3 XGBoost classifiers on synthetic clinical data:
  1. readmission_30d  — 30-day hospital readmission risk
  2. mortality_30d    — 30-day mortality risk
  3. complication     — complication risk during treatment

Honest framing (from spec/strategy doc):
  - Uses synthetic data generated here (not MIMIC-III, which requires credentials)
  - Designed to produce plausible risk scores for the demo, not production accuracy
  - Architecture is correct; real deployment would train on population-scale EHR data

Usage:
    cd agents/digital_twin
    python train_models.py

Output:
    models/readmission_30d.json
    models/mortality_30d.json
    models/complication.json
    models/feature_names.json   ← feature order for inference
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

np.random.seed(42)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_SAMPLES = 8_000   # at 8% mortality → ~640 positives in train, plenty to learn


# ── Feature definitions (must match feature_engineering.py exactly) ────────────
FEATURE_NAMES = [
    "age",
    "gender_male",
    "wbc",
    "creatinine",
    "albumin",
    "glucose",
    "crp",
    "hemoglobin",
    "potassium",
    "comorbidity_count",
    "has_diabetes",
    "has_ckd",
    "has_chf",
    "has_copd",
    "has_atrial_fibrillation",
    "med_count",
    "critical_lab_count",
    "on_anticoagulant",
    "on_steroid",
]

N_FEATURES = len(FEATURE_NAMES)


def generate_synthetic_patients(n: int) -> pd.DataFrame:
    """
    Generate clinically plausible synthetic patient feature vectors.
    Distributions are based on typical hospital admission populations.
    """
    age = np.random.normal(65, 15, n).clip(18, 95)
    gender_male = np.random.binomial(1, 0.48, n)
    wbc = np.random.lognormal(np.log(9), 0.4, n).clip(1, 40)
    creatinine = np.random.lognormal(np.log(1.1), 0.4, n).clip(0.4, 8)
    albumin = np.random.normal(3.8, 0.6, n).clip(1.5, 5.5)
    glucose = np.random.normal(120, 40, n).clip(60, 500)
    crp = np.random.lognormal(np.log(30), 1.2, n).clip(1, 300)
    hemoglobin = np.random.normal(12.5, 2.0, n).clip(5, 18)
    potassium = np.random.normal(4.0, 0.5, n).clip(2.5, 7.0)
    comorbidity_count = np.random.poisson(2.5, n).clip(0, 8)
    has_diabetes = np.random.binomial(1, 0.35, n)
    has_ckd = np.random.binomial(1, 0.25, n)
    has_chf = np.random.binomial(1, 0.18, n)
    has_copd = np.random.binomial(1, 0.20, n)
    has_af = np.random.binomial(1, 0.22, n)
    med_count = np.random.poisson(5, n).clip(0, 15)
    critical_lab_count = np.random.poisson(0.8, n).clip(0, 5)
    on_anticoagulant = np.random.binomial(1, 0.25, n)
    on_steroid = np.random.binomial(1, 0.15, n)

    df = pd.DataFrame({
        "age": age,
        "gender_male": gender_male,
        "wbc": wbc,
        "creatinine": creatinine,
        "albumin": albumin,
        "glucose": glucose,
        "crp": crp,
        "hemoglobin": hemoglobin,
        "potassium": potassium,
        "comorbidity_count": comorbidity_count,
        "has_diabetes": has_diabetes,
        "has_ckd": has_ckd,
        "has_chf": has_chf,
        "has_copd": has_copd,
        "has_atrial_fibrillation": has_af,
        "med_count": med_count,
        "critical_lab_count": critical_lab_count,
        "on_anticoagulant": on_anticoagulant,
        "on_steroid": on_steroid,
    })

    return df


def generate_labels(df: pd.DataFrame) -> dict:
    n = len(df)

    # ── 30-day readmission ── target ~20%
    readmit_logit = (
        -3.3
        + 0.03  * (df["age"] - 65)
        + 0.8   * df["has_ckd"]
        + 1.0   * df["has_chf"]
        + 0.6   * df["has_copd"]
        + 0.4   * df["has_diabetes"]
        + 0.20  * df["comorbidity_count"]
        + 0.5   * (df["creatinine"] - 1.0).clip(0)
        + 0.3   * df["critical_lab_count"]
        - 0.3   * (df["albumin"] - 3.5).clip(None, 0)
        + 0.4   * df["on_steroid"]           # ← add: steroids → readmission risk
        + 0.3   * (df["hemoglobin"] < 10).astype(int)  # ← add: anemia → readmission
        + np.random.normal(0, 0.3, n)   # was 0.4
    )
    readmit_prob = 1 / (1 + np.exp(-readmit_logit))
    y_readmit = (readmit_prob > np.random.uniform(0, 1, n)).astype(int)

    
    # ── 30-day mortality ──
    mort_logit = (
        -3.8
        + 0.05  * (df["age"] - 65)       # was 0.04
        + 1.0   * df["has_chf"]          # was 0.7
        + 0.7   * df["has_ckd"]          # was 0.5
        + 0.9   * (df["creatinine"] > 2.0).astype(int)   # was 0.6
        + 0.7   * df["critical_lab_count"]               # was 0.5
        + 0.8   * (df["albumin"] < 3.0).astype(int)      # was 0.6
        + 0.5   * (df["wbc"] > 15).astype(int)
        + np.random.normal(0, 0.4, n)    # was 0.6
    )
    mort_prob = 1 / (1 + np.exp(-mort_logit))
    y_mort = (mort_prob > np.random.uniform(0, 1, n)).astype(int)

    # ── Complication ──
    comp_logit = (
        -3.4
        + 0.7   * df["has_diabetes"]     # was 0.5
        + 0.6   * df["on_anticoagulant"] # was 0.4
        + 0.7   * df["has_ckd"]          # was 0.5
        + 0.5   * df["on_steroid"]       # was 0.4
        + 0.15  * df["med_count"]        # was 0.12
        + 0.4   * df["critical_lab_count"]
        + 0.04  * (df["age"] - 65)
        + np.random.normal(0, 0.4, n)    # was 0.6
    )
    comp_prob = 1 / (1 + np.exp(-comp_logit))
    y_comp = (comp_prob > np.random.uniform(0, 1, n)).astype(int)

    return {
        "readmission_30d": y_readmit,
        "mortality_30d":   y_mort,
        "complication":    y_comp,
        "readmission_90d": _generate_readmission_90d(df, y_readmit, n),
        "mortality_1yr":   _generate_mortality_1yr(df, y_mort, n),
    }


def _generate_readmission_90d(df: pd.DataFrame, y_readmit_30d: np.ndarray, n: int) -> np.ndarray:
    """
    90-day readmission: higher base rate (~35%) than 30d (~20%).
    Patients who readmit at 30d almost always also count at 90d.
    """
    readmit_90d_logit = (
        -2.6
        + 0.03  * (df["age"] - 65)
        + 0.9   * df["has_ckd"]
        + 1.1   * df["has_chf"]
        + 0.7   * df["has_copd"]
        + 0.5   * df["has_diabetes"]
        + 0.25  * df["comorbidity_count"]
        + 0.6   * (df["creatinine"] - 1.0).clip(0)
        + 0.4   * df["critical_lab_count"]
        + 0.5   * df["on_steroid"]
        + 0.4   * (df["hemoglobin"] < 10).astype(int)
        + np.random.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-readmit_90d_logit))
    # Ensure 30d readmitters are also 90d readmitters
    y = (prob > np.random.uniform(0, 1, n)).astype(int)
    y = np.maximum(y, y_readmit_30d)
    return y


def _generate_mortality_1yr(df: pd.DataFrame, y_mort_30d: np.ndarray, n: int) -> np.ndarray:
    """
    1-year mortality: ~15% base rate.
    30-day non-survivors are by definition also 1-year non-survivors.
    """
    mort_1yr_logit = (
        -3.0
        + 0.06  * (df["age"] - 65)
        + 1.1   * df["has_chf"]
        + 0.8   * df["has_ckd"]
        + 1.0   * (df["creatinine"] > 2.0).astype(int)
        + 0.8   * df["critical_lab_count"]
        + 0.9   * (df["albumin"] < 3.0).astype(int)
        + 0.6   * (df["wbc"] > 15).astype(int)
        + 0.5   * df["has_copd"]
        + np.random.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-mort_1yr_logit))
    y = (prob > np.random.uniform(0, 1, n)).astype(int)
    # 30-day deaths are also 1-year deaths
    y = np.maximum(y, y_mort_30d)
    return y


def train_and_save_model(
    X_train, y_train,
    X_test,  y_test,
    name: str,
) -> xgb.XGBClassifier:
    """Train a single XGBoost classifier and save it."""
    model = xgb.XGBClassifier(
        n_estimators=300,          # was 120
        max_depth=3,               # was 4 — shallower = less overfit on small data
        learning_rate=0.05,        # was 0.08 — slower = more stable
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,        # was 3 — more conservative on rare positives
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        reg_alpha=0.1,             # L1 — add this
        reg_lambda=1.5,            # L2 — increase this
        eval_metric="auc",         # was logloss — optimize directly for AUC
        early_stopping_rounds=20,  # add this — prevents overfit
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    prevalence = y_train.mean()
    print(f"  {name}: AUC={auc:.3f}, prevalence={prevalence:.1%}")

    # Save
    out_path = MODELS_DIR / f"{name}.json"
    model.save_model(str(out_path))
    print(f"  Saved → {out_path}")
    return model


def main():
    print("=" * 55)
    print("Digital Twin — Risk Model Training")
    print("=" * 55)
    print(f"\nGenerating {N_SAMPLES} synthetic patients...")

    df = generate_synthetic_patients(N_SAMPLES)
    labels = generate_labels(df)

    X = df[FEATURE_NAMES].values

    print("\nTraining 3 XGBoost risk classifiers...")
    print(f"Features: {N_FEATURES} | Train/Test: 80/20 split")
    print()

    for target_name, y in labels.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_and_save_model(X_train, y_train, X_test, y_test, target_name)

    # Save feature names so inference.py loads them in the right order
    feat_path = MODELS_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"\n  Feature names saved → {feat_path}")

    print("\n" + "=" * 55)
    print("✅ All 3 models trained and saved!")
    print("   Models trained on synthetic data — representative, not production-scale.")
    print("   Real deployment would train on population-scale EHR data (MIMIC-IV etc.).")
    print("=" * 55)


if __name__ == "__main__":
    main()