# ============================================================
# MODEL AUDIT DIAGNOSTICS
# Bank Stress Testing Simulator
# ============================================================
# Reproduces the exact baseline train/test pipeline for all
# three models and audits:
#
#   1. Train vs test metrics (incl. AUC-ROC, PR-AUC, Brier)
#   2. Group-aware cross-validation
#   3. Overfitting severity in XGBoost / Random Forest
#   4. Root-cause experiments (class weighting, depth limits)
#   5. Linearity diagnostics of the data-generating process
#
# Output: outputs/model_audit_results.xlsx
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# ============================================================
# 1. PATHS AND CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "processed" / "bank_health_features.csv"
PANEL_FILE = BASE_DIR / "processed" / "bank_stress_simulated_panel_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "model_audit_results.xlsx"

OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_ORDER = ["Healthy", "Stressed", "Critical"]

TARGET_MAPPING = {"Healthy": 0, "Stressed": 1, "Critical": 2}

FEATURE_COLS = [
    "size_score",
    "concentration_flag",
    "top_sector_weight",
    "sector_risk_score",
    "bank_risk_factor",
    "loan_to_asset_ratio",
    "deposit_to_asset_ratio",
    "car_buffer",
    "liquidity_buffer",
    "baseline_roa_pct",
    "severity_score",
    "shock_severity_score",
    "concentration_x_severity",
    "risk_x_severity",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

features = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("MODEL AUDIT DIAGNOSTICS")
print("=" * 70)

print("\nDataset shape:", features.shape)

X = features[FEATURE_COLS].copy()
y = features["bank_condition"].copy()
y_encoded = features["bank_condition"].map(TARGET_MAPPING)
groups = features["scenario_id"]


# ============================================================
# 3. CLASS DISTRIBUTION AND MISSING VALUES
# ============================================================

print("\nTarget distribution (full data):")
print(y.value_counts(normalize=True).round(4) * 100)

print("\nMissing values per feature:")
print(X.isna().sum()[X.isna().sum() > 0])


# ============================================================
# 4. REPRODUCE THE EXACT BASELINE SPLIT
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()
y_train_enc = y_encoded.iloc[train_idx].copy()
y_test_enc = y_encoded.iloc[test_idx].copy()

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Train scenarios:", groups.iloc[train_idx].nunique())
print("Test scenarios :", groups.iloc[test_idx].nunique())

print("\nTrain class distribution:")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nTest class distribution:")
print((y_test.value_counts(normalize=True) * 100).round(2))

overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print("\nScenario overlap (must be empty):", overlap)


# ============================================================
# 5. METRIC HELPERS
# ============================================================

def one_hot(y_true):
    return np.column_stack([
        (y_true == c).astype(int).values
        for c in CLASS_ORDER
    ])


def evaluate(y_true, y_pred, y_proba, proba_classes):
    """y_proba columns follow proba_classes order.

    Ranking metrics use integer labels (0, 1, 2) because
    sklearn requires ordered labels for multiclass AUC
    and log loss.
    """

    proba = reorder_proba(y_proba, proba_classes)

    Y_true = one_hot(y_true)
    y_true_int = y_true.map(TARGET_MAPPING).values

    report = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(
            y_true, y_pred,
            average="macro",
            labels=CLASS_ORDER,
            zero_division=0
        ),
        "Critical Recall": (
            (y_pred[y_true == "Critical"] == "Critical").mean()
            if (y_true == "Critical").any() else np.nan
        ),
        "AUC-ROC (macro OVR)": roc_auc_score(
            y_true_int, proba,
            multi_class="ovr",
            average="macro",
            labels=[0, 1, 2]
        ),
        "PR-AUC (macro OVR)": np.mean([
            average_precision_score(
                (y_true == c).astype(int),
                proba[:, i]
            )
            for i, c in enumerate(CLASS_ORDER)
        ]),
        "PR-AUC (Critical)": average_precision_score(
            (y_true == "Critical").astype(int),
            proba[:, CLASS_ORDER.index("Critical")]
        ),
        "Brier (multiclass)": np.mean(
            np.sum((proba - Y_true) ** 2, axis=1)
        ),
        "Log Loss": log_loss(
            y_true_int, proba, labels=[0, 1, 2]
        ),
        "P(max prob >= 0.9)": float(
            (proba.max(axis=1) >= 0.9).mean()
        ),
    }

    return report


def reorder_proba(y_proba, proba_classes):
    """Reorder probability columns to CLASS_ORDER."""

    idx = [proba_classes.index(c) for c in CLASS_ORDER]
    return np.asarray(y_proba)[:, idx]


def fit_predict(model, use_encoded=False, sample_weight=None):
    """Fit model on the train split, return train/test predictions."""

    y_tr = y_train_enc if use_encoded else y_train
    y_te = y_test_enc if use_encoded else y_test

    if sample_weight is not None:
        model.fit(X_train, y_tr, classifier__sample_weight=sample_weight) \
            if isinstance(model, Pipeline) else model.fit(
                X_train, y_tr, sample_weight=sample_weight
            )
    else:
        model.fit(X_train, y_tr)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    proba_train = model.predict_proba(X_train)
    proba_test = model.predict_proba(X_test)

    classes = list(model.classes_)

    if use_encoded:
        reverse = {0: "Healthy", 1: "Stressed", 2: "Critical"}
        pred_train = pd.Series(pred_train).map(reverse).values
        pred_test = pd.Series(pred_test).map(reverse).values
        classes = [reverse[c] for c in classes]

    return pred_train, pred_test, proba_train, proba_test, classes


def audit_model(name, model, use_encoded=False, sample_weight=None):
    """Full train/test audit for one model configuration."""

    pred_tr, pred_te, proba_tr, proba_te, classes = fit_predict(
        model, use_encoded=use_encoded, sample_weight=sample_weight
    )

    train_metrics = evaluate(y_train, pred_tr, proba_tr, classes)
    test_metrics = evaluate(y_test, pred_te, proba_te, classes)

    row = {"Model": name}

    for k, v in train_metrics.items():
        row["Train " + k] = round(v, 4)

    for k, v in test_metrics.items():
        row["Test " + k] = round(v, 4)

    for k in train_metrics:
        row["Gap " + k] = round(
            train_metrics[k] - test_metrics[k], 4
        )

    return row


# ============================================================
# 6. MODEL FACTORIES (exact baseline replicas + variants)
# ============================================================

def make_logistic(class_weight="balanced"):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=42
        )),
    ])


def make_random_forest(
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced"
):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )),
    ])


def make_xgboost(max_depth=6, min_child_weight=2):
    return XGBClassifier(
        n_estimators=500,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=min_child_weight,
        gamma=0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )


balanced_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_enc
)


# ============================================================
# 7. AUDIT ALL MODEL CONFIGURATIONS
# ============================================================

model_specs = [
    ("Logistic Regression (baseline)",
     make_logistic(), False, None),
    ("Random Forest (baseline)",
     make_random_forest(), False, None),
    ("XGBoost (baseline)",
     make_xgboost(), True, None),
    ("XGBoost + class weights",
     make_xgboost(), True, balanced_weights),
    ("XGBoost + depth 3 + mcw 10",
     make_xgboost(max_depth=3, min_child_weight=10), True, None),
    ("XGBoost + class weights + depth 3",
     make_xgboost(max_depth=3, min_child_weight=10), True,
     balanced_weights),
    ("Random Forest + max_depth 8",
     make_random_forest(max_depth=8), False, None),
    ("Random Forest + depth 8 + leaf 10",
     make_random_forest(max_depth=8, min_samples_leaf=10),
     False, None),
    ("Logistic Regression (no weights)",
     make_logistic(class_weight=None), False, None),
]

audit_rows = []

for name, model, use_enc, sw in model_specs:
    print(f"\nAuditing: {name}")
    audit_rows.append(audit_model(name, model, use_enc, sw))

audit_table = pd.DataFrame(audit_rows)

print("\n" + "=" * 70)
print("TRAIN / TEST AUDIT TABLE")
print("=" * 70)

key_cols = [
    "Model",
    "Train Accuracy", "Test Accuracy",
    "Train Balanced Acc", "Test Balanced Acc",
    "Train Macro F1", "Test Macro F1",
    "Train Critical Recall", "Test Critical Recall",
]

print(audit_table[key_cols].to_string(index=False))

print("\n" + "=" * 70)
print("PROBABILISTIC METRICS (TEST)")
print("=" * 70)

prob_cols = [
    "Model",
    "Train AUC-ROC (macro OVR)", "Test AUC-ROC (macro OVR)",
    "Train PR-AUC (macro OVR)", "Test PR-AUC (macro OVR)",
    "Train PR-AUC (Critical)", "Test PR-AUC (Critical)",
    "Train Brier (multiclass)", "Test Brier (multiclass)",
    "Train Log Loss", "Test Log Loss",
    "Test P(max prob >= 0.9)",
]

print(audit_table[prob_cols].to_string(index=False))

print("\n" + "=" * 70)
print("OVERFITTING GAPS (TRAIN - TEST)")
print("=" * 70)

gap_cols = ["Model"] + [
    c for c in audit_table.columns
    if c.startswith("Gap ")
]

print(audit_table[gap_cols].to_string(index=False))


# ============================================================
# 8. GROUP-AWARE CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GROUP-AWARE CROSS-VALIDATION (StratifiedGroupKFold, 5 folds)")
print("=" * 70)

cv_specs = [
    ("Logistic Regression (baseline)",
     make_logistic(), False, None),
    ("Random Forest (baseline)",
     make_random_forest(), False, None),
    ("XGBoost (baseline)",
     make_xgboost(), True, None),
    ("XGBoost + class weights",
     make_xgboost(), True, None),
    ("Random Forest + max_depth 8",
     make_random_forest(max_depth=8), False, None),
]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_rows = []

for name, model, use_enc, _ in cv_specs:

    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(
        cv.split(X, y, groups=groups)
    ):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y_encoded.iloc[tr_idx] if use_enc \
            else y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        fold_weights = compute_sample_weight(
            class_weight="balanced", y=y_tr
        ) if name.endswith("class weights") else None

        if fold_weights is not None:
            model.fit(X_tr, y_tr, sample_weight=fold_weights)
        else:
            model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)

        classes = list(model.classes_)
        reverse = {
            0: "Healthy", 1: "Stressed", 2: "Critical"
        }

        if use_enc:
            pred = pd.Series(pred).map(reverse).values
            classes = [reverse[c] for c in classes]

        proba = reorder_proba(proba, classes)

        fold_metrics.append({
            "Balanced Acc": balanced_accuracy_score(y_te, pred),
            "Macro F1": f1_score(
                y_te, pred, average="macro",
                labels=CLASS_ORDER, zero_division=0
            ),
            "Critical Recall": (
                pred[y_te.values == "Critical"] == "Critical"
            ).mean(),
            "AUC-ROC": roc_auc_score(
                y_te.map(TARGET_MAPPING).values, proba,
                multi_class="ovr",
                average="macro", labels=[0, 1, 2]
            ),
            "Brier": np.mean(
                np.sum(
                    (proba - one_hot(y_te)) ** 2, axis=1
                )
            ),
        })

    folds_df = pd.DataFrame(fold_metrics)

    cv_rows.append({
        "Model": name,
        "Balanced Acc (mean)": round(
            folds_df["Balanced Acc"].mean(), 4),
        "Balanced Acc (std)": round(
            folds_df["Balanced Acc"].std(), 4),
        "Macro F1 (mean)": round(
            folds_df["Macro F1"].mean(), 4),
        "Macro F1 (std)": round(
            folds_df["Macro F1"].std(), 4),
        "Critical Recall (mean)": round(
            folds_df["Critical Recall"].mean(), 4),
        "Critical Recall (std)": round(
            folds_df["Critical Recall"].std(), 4),
        "AUC-ROC (mean)": round(
            folds_df["AUC-ROC"].mean(), 4),
        "AUC-ROC (std)": round(
            folds_df["AUC-ROC"].std(), 4),
        "Brier (mean)": round(
            folds_df["Brier"].mean(), 4),
    })

cv_table = pd.DataFrame(cv_rows)

print(cv_table.to_string(index=False))


# ============================================================
# 9. LINEARITY DIAGNOSTICS OF THE DATA-GENERATING PROCESS
# ============================================================

print("\n" + "=" * 70)
print("LINEARITY DIAGNOSTICS")
print("=" * 70)

panel = pd.read_csv(PANEL_FILE)

panel_cols = [
    "bank_id", "scenario_id",
    "car_after_pct", "liquidity_after_pct", "roa_after_pct"
]

merged = features.merge(
    panel[panel_cols],
    on=["bank_id", "scenario_id"],
    how="left"
)

print("\nPost-shock CAR by condition class:")

car_by_condition = merged.groupby("bank_condition")[
    "car_after_pct"
].agg(["min", "mean", "max"]).round(3)

print(car_by_condition)

print("\nPost-shock liquidity by condition class:")

liq_by_condition = merged.groupby("bank_condition")[
    "liquidity_after_pct"
].agg(["min", "mean", "max"]).round(3)

print(liq_by_condition)

# --- OLS: how linear is the simulation outcome in the features?

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

lin_rows = []

for target_col in [
    "car_after_pct",
    "liquidity_after_pct",
    "bank_condition_code"
]:
    target = merged[target_col]
    valid = ~target.isna()

    ols = LinearRegression().fit(
        X_imputed[valid.values],
        target[valid]
    )

    r2 = ols.score(X_imputed[valid.values], target[valid])

    lin_rows.append({
        "Target": target_col,
        "Linear R2 on 14 features": round(r2, 4),
        "Rows used": int(valid.sum()),
    })

linearity_table = pd.DataFrame(lin_rows)

print("\nOLS linear fit of simulation outputs on the 14 features:")
print(linearity_table.to_string(index=False))

# --- Spearman correlation of features with the ordinal target

spearman_rows = []

for col in FEATURE_COLS:
    rho = merged[[col, "bank_condition_code"]].corr(
        method="spearman"
    ).iloc[0, 1]
    spearman_rows.append({
        "Feature": col,
        "Spearman rho vs target": round(rho, 4)
    })

spearman_table = pd.DataFrame(spearman_rows).sort_values(
    by="Spearman rho vs target",
    key=abs,
    ascending=False
).reset_index(drop=True)

print("\nSpearman correlation of features with target:")
print(spearman_table.to_string(index=False))


# ============================================================
# 10. FEATURE IMPORTANCE COMPARISON (TOP 5 PER MODEL)
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE COMPARISON (top 5)")
print("=" * 70)

lr_pipeline = make_logistic()
lr_pipeline.fit(X_train, y_train)

lr_coefs = lr_pipeline.named_steps[
    "classifier"
].coef_

lr_importance = pd.Series(
    np.abs(lr_coefs).mean(axis=0),
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\nLogistic Regression (mean |coef|, standardized):")
print(lr_importance.head(5).round(4).to_string())

rf_pipeline = make_random_forest()
rf_pipeline.fit(X_train, y_train)

rf_importance = pd.Series(
    rf_pipeline.named_steps["classifier"].feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\nRandom Forest (impurity importance):")
print(rf_importance.head(5).round(4).to_string())

xgb_model = make_xgboost()
xgb_model.fit(X_train, y_train_enc)

xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\nXGBoost (gain importance):")
print(xgb_importance.head(5).round(4).to_string())

importance_table = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "LR |coef| (mean)": [
        round(lr_importance[f], 4) for f in FEATURE_COLS
    ],
    "RF importance": [
        round(rf_importance[f], 4) for f in FEATURE_COLS
    ],
    "XGB importance": [
        round(xgb_importance[f], 4) for f in FEATURE_COLS
    ],
}).sort_values(
    by="RF importance", ascending=False
).reset_index(drop=True)

# Tree depth statistics for the Random Forest

rf_est = rf_pipeline.named_steps["classifier"]

rf_depths = [
    est.tree_.max_depth for est in rf_est.estimators_
]

print("\nRandom Forest tree depth stats:")
print(
    "max depth across trees:",
    max(rf_depths),
    "| mean depth:",
    round(np.mean(rf_depths), 1)
)

# Train accuracy of the fully-grown Random Forest

rf_train_pred = rf_pipeline.predict(X_train)
print(
    "Random Forest TRAIN accuracy:",
    round(accuracy_score(y_train, rf_train_pred), 4)
)


# ============================================================
# 11. PREPROCESSING LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING LEAKAGE CHECK")
print("=" * 70)

# shock_severity_score used full-dataset min-max scaling
# in feature_engineering.ipynb (before the split).

shock_cols = [
    "gdp_shock_pp", "unemp_shock_pp", "rate_shock_pp",
    "credit_spread_bps", "inflation_shock_pp",
    "fx_devaluation_pct"
]

panel_full = panel.merge(
    features[["bank_id", "scenario_id"]].drop_duplicates(),
    on=["bank_id", "scenario_id"], how="inner"
)

full_mins = {
    c: panel_full[c].abs().min() for c in shock_cols
}
full_maxs = {
    c: panel_full[c].abs().max() for c in shock_cols
}

train_scenarios = set(groups.iloc[train_idx])
train_mask = panel_full["scenario_id"].isin(train_scenarios)

train_mins = {
    c: panel_full.loc[train_mask, c].abs().min()
    for c in shock_cols
}
train_maxs = {
    c: panel_full.loc[train_mask, c].abs().max()
    for c in shock_cols
}

diff_rows = []

for c in shock_cols:
    diff_rows.append({
        "Shock column": c,
        "Full-data min": round(full_mins[c], 4),
        "Train-only min": round(train_mins[c], 4),
        "Full-data max": round(full_maxs[c], 4),
        "Train-only max": round(train_maxs[c], 4),
    })

leakage_table = pd.DataFrame(diff_rows)

print("\nMin-max scaling inputs (full data vs train only):")
print(leakage_table.to_string(index=False))


# ============================================================
# 12. VERIFY REPORTED BASELINE METRICS (from outputs/)
# ============================================================

print("\n" + "=" * 70)
print("REPORTED BASELINE METRICS (outputs/ Excel files)")
print("=" * 70)

reported_rows = []

for fname, label in [
    ("logistic_regression_results.xlsx", "Logistic Regression"),
    ("random_forest_results.xlsx", "Random Forest"),
    ("xgboost_results.xlsx", "XGBoost"),
]:
    path = OUTPUT_DIR / fname

    if path.exists():
        m = pd.read_excel(path, sheet_name="Metrics")
        row = {"Model": label}

        for _, r in m.iterrows():
            row[r["Metric"]] = round(r["Score"], 4)

        reported_rows.append(row)

if reported_rows:
    reported_table = pd.DataFrame(reported_rows)
    print(reported_table.to_string(index=False))
else:
    reported_table = pd.DataFrame()
    print("No baseline result files found.")


# ============================================================
# 13. SAVE AUDIT RESULTS
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    audit_table.to_excel(
        writer, sheet_name="Train Test Audit", index=False
    )

    cv_table.to_excel(
        writer, sheet_name="Cross Validation", index=False
    )

    linearity_table.to_excel(
        writer, sheet_name="Linearity Diagnostics", index=False
    )

    spearman_table.to_excel(
        writer, sheet_name="Feature Target Correlation",
        index=False
    )

    importance_table.to_excel(
        writer, sheet_name="Feature Importance", index=False
    )

    leakage_table.to_excel(
        writer, sheet_name="Scaling Leakage Check", index=False
    )

    if not reported_table.empty:
        reported_table.to_excel(
            writer, sheet_name="Reported Baselines", index=False
        )

print("\nAudit results saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("MODEL AUDIT COMPLETED")
print("=" * 70)
