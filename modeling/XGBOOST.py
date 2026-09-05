# ============================================================
# XGBOOST BASELINE MODEL
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# 1. LOAD FEATURE DATASET
# ============================================================

features = pd.read_csv(
    "bank-stress-testing-simulator/processed/bank_health_features.csv"
)

print("Dataset shape:", features.shape)

print("\nColumns:")
print(features.columns.tolist())


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "bank_condition"

y = features[TARGET]


# ============================================================
# 3. SELECT MODEL FEATURES
# ============================================================
# IDs are identifiers, NOT predictive features.
#
# bank_condition_code is a numeric encoding of the target,
# so it must NOT be used as a predictor.
# ============================================================

DROP_COLS = [
    "bank_id",
    "scenario_id",
    "bank_condition",
    "bank_condition_code"
]

X = features.drop(columns=DROP_COLS)

print("\nModel features:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 5. MISSING VALUE INFORMATION
# ============================================================
# Missing values will be handled using median imputation.
#
# We do NOT modify the original feature dataset.
# ============================================================

missing = X.isna().sum()

print("\nMissing values:")
print(missing[missing > 0])


# ============================================================
# 6. CONVERT TARGET TO NUMERIC LABELS
# ============================================================
# XGBoost multiclass classification requires numeric labels.
#
# Healthy  -> 0
# Stressed -> 1
# Critical -> 2
# ============================================================

target_mapping = {
    "Healthy": 0,
    "Stressed": 1,
    "Critical": 2
}

y_encoded = y.map(target_mapping)

print("\nTarget encoding:")
print(target_mapping)


# ============================================================
# 7. SCENARIO-AWARE TRAIN / TEST SPLIT
# ============================================================
# The same scenario must not appear in both train and test.
# ============================================================

groups = features["scenario_id"]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y_encoded,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y_encoded.iloc[train_idx].copy()
y_test = y_encoded.iloc[test_idx].copy()


print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

print(
    "\nTraining scenarios:",
    features.iloc[train_idx]["scenario_id"].nunique()
)

print(
    "Testing scenarios :",
    features.iloc[test_idx]["scenario_id"].nunique()
)


# ============================================================
# 8. VERIFY NO SCENARIO LEAKAGE
# ============================================================

train_scenarios = set(
    features.iloc[train_idx]["scenario_id"]
)

test_scenarios = set(
    features.iloc[test_idx]["scenario_id"]
)

overlap = train_scenarios.intersection(
    test_scenarios
)

print("\nScenario overlap:", overlap)

if len(overlap) == 0:
    print("✓ No scenario leakage detected.")
else:
    print("⚠ WARNING: Scenario leakage detected!")


# ============================================================
# 9. MEDIAN IMPUTATION
# ============================================================
# Imputer is fitted ONLY on training data.
# This prevents information from the test set entering training.
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)

print("\n✓ Missing values handled using median imputation.")


# ============================================================
# 10. XGBOOST MODEL
# ============================================================
# Baseline XGBoost configuration.
#
# This is the same configuration used in our
# original baseline experiment.
# ============================================================

xgb_model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=2,

    gamma=0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    objective="multi:softprob",

    num_class=3,

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train_imputed,
    y_train
)

print("✓ XGBoost training completed.")


# ============================================================
# 12. GENERATE PREDICTIONS
# ============================================================

y_pred_encoded = xgb_model.predict(
    X_test_imputed
)

y_proba = xgb_model.predict_proba(
    X_test_imputed
)

print("\nPredictions generated.")


# ============================================================
# 13. CONVERT PREDICTIONS BACK TO ORIGINAL LABELS
# ============================================================

reverse_mapping = {
    0: "Healthy",
    1: "Stressed",
    2: "Critical"
}

y_pred = pd.Series(
    y_pred_encoded
).map(reverse_mapping).values

y_test_labels = y_test.map(
    reverse_mapping
).values


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_labels,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test_labels,
    y_pred
)

macro_f1 = f1_score(
    y_test_labels,
    y_pred,
    average="macro"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

class_order = [
    "Healthy",
    "Stressed",
    "Critical"
]

report = classification_report(
    y_test_labels,
    y_pred,
    labels=class_order,
    output_dict=True
)

report_output = pd.DataFrame(
    report
).transpose()


# ============================================================
# 16. CRITICAL CLASS RECALL
# ============================================================

critical_recall = report[
    "Critical"
]["recall"]


# ============================================================
# 17. PRINT MODEL RESULTS
# ============================================================

print("\n==========================================")
print("XGBOOST RESULTS")
print("==========================================")

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.4f}"
)

print(
    f"Macro F1          : {macro_f1:.4f}"
)

print(
    f"Critical Recall   : {critical_recall:.4f}"
)


# ============================================================
# 18. PRINT CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test_labels,
        y_pred,
        labels=class_order,
        digits=4
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_labels,
    y_pred,
    labels=class_order
)

cm_output = pd.DataFrame(
    cm,
    index=[
        f"Actual_{c}"
        for c in class_order
    ],
    columns=[
        f"Predicted_{c}"
        for c in class_order
    ]
)

print("\nConfusion Matrix:")
print(cm_output)


# ============================================================
# 20. PREDICTIONS + PROBABILITIES
# ============================================================

predictions_output = pd.DataFrame({

    "actual_condition":
        y_test_labels,

    "predicted_condition":
        y_pred,

    "prob_healthy":
        y_proba[:, 0],

    "prob_stressed":
        y_proba[:, 1],

    "prob_critical":
        y_proba[:, 2]
})


print("\nSample probability predictions:")
print(
    predictions_output.head(10)
)


# ============================================================
# 21. MODEL METRICS TABLE
# ============================================================

metrics_output = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Critical Recall"
    ],

    "Score": [
        accuracy,
        balanced_accuracy,
        macro_f1,
        critical_recall
    ]
})


# ============================================================
# 22. XGBOOST FEATURE IMPORTANCE
# ============================================================
# Feature importance is extracted from the trained
# XGBoost model.
# ============================================================

feature_importance_output = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        xgb_model.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 23. SAVE ALL RESULTS IN ONE EXCEL FILE
# ============================================================
# Only ONE output file is created.
#
# Excel sheets:
#   1. Metrics
#   2. Classification Report
#   3. Confusion Matrix
#   4. Predictions
#   5. Feature Importance
# ============================================================

output_file = (
    "xgboost_results.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    metrics_output.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
    )

    report_output.to_excel(
        writer,
        sheet_name="Classification Report"
    )

    cm_output.to_excel(
        writer,
        sheet_name="Confusion Matrix"
    )

    predictions_output.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    feature_importance_output.to_excel(
        writer,
        sheet_name="Feature Importance",
        index=False
    )


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("XGBOOST OUTPUT SAVED")
print("==========================================")

print(
    f"Output file: {output_file}"
)

print("\nExcel sheets:")

print("1. Metrics")
print("2. Classification Report")
print("3. Confusion Matrix")
print("4. Predictions")
print("5. Feature Importance")

print("\noutput file created.")
print("✓ XGBoost completed successfully.")