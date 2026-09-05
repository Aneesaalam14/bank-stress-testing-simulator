# ============================================================
# LOGISTIC REGRESSION BASELINE MODEL
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
# Missing values are handled automatically inside the pipeline
# using median imputation.
#
# We only display the columns containing missing values.
# We do NOT modify the original feature dataset.
# ============================================================

missing = X.isna().sum()

print("\nMissing values:")
print(missing[missing > 0])


# ============================================================
# 6. SCENARIO-AWARE TRAIN / TEST SPLIT
# ============================================================
# The same scenario must not appear in both train and test.
#
# This is important because our dataset is a
# Bank × Scenario panel.
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
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


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
# 7. VERIFY NO SCENARIO LEAKAGE
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
# 8. LOGISTIC REGRESSION PIPELINE
# ============================================================
# Step 1 → Median imputation
# Step 2 → Standard scaling
# Step 3 → Balanced Logistic Regression
#
# All preprocessing is learned from the training data only.
# ============================================================

logistic_model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

print("✓ Logistic Regression training completed.")


# ============================================================
# 10. GENERATE PREDICTIONS
# ============================================================

y_pred = logistic_model.predict(
    X_test
)

y_proba = logistic_model.predict_proba(
    X_test
)

print("\nPredictions generated.")


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

class_order = [
    "Healthy",
    "Stressed",
    "Critical"
]

report = classification_report(
    y_test,
    y_pred,
    labels=class_order,
    output_dict=True
)

report_output = pd.DataFrame(
    report
).transpose()


# ============================================================
# 13. CRITICAL CLASS RECALL
# ============================================================
# In banking stress testing, identifying Critical banks
# correctly is particularly important.
# ============================================================

critical_recall = report["Critical"]["recall"]


# ============================================================
# 14. PRINT MODEL RESULTS
# ============================================================

print("\n==========================================")
print("LOGISTIC REGRESSION RESULTS")
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
# 15. PRINT CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=class_order,
        digits=4
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
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
# 17. PREDICTIONS + PROBABILITIES
# ============================================================

classes = list(
    logistic_model.named_steps[
        "classifier"
    ].classes_
)

predictions_output = pd.DataFrame({

    "actual_condition":
        y_test.values,

    "predicted_condition":
        y_pred,

    "prob_healthy":
        y_proba[
            :,
            classes.index("Healthy")
        ],

    "prob_stressed":
        y_proba[
            :,
            classes.index("Stressed")
        ],

    "prob_critical":
        y_proba[
            :,
            classes.index("Critical")
        ]
})


print("\nSample probability predictions:")
print(
    predictions_output.head(10)
)


# ============================================================
# 18. MODEL METRICS TABLE
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
# 19. SAVE ALL RESULTS IN ONE EXCEL FILE
# ============================================================
# Only ONE output file is created.
#
# Excel sheets:
#   1. Metrics
#   2. Classification Report
#   3. Confusion Matrix
#   4. Predictions
# ============================================================

output_file = (
    "logistic_regression_results.xlsx"
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


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("LOGISTIC REGRESSION OUTPUT SAVED")
print("==========================================")

print(
    f"Output file: {output_file}"
)

print("\nExcel sheets:")

print("1. Metrics")
print("2. Classification Report")
print("3. Confusion Matrix")
print("4. Predictions")

print("\noutput file created.")
print("✓ Logistic Regression completed successfully.")