# ============================================================
# BANK STRESS TESTING SIMULATOR
# STAGE 9: RISK SCORING
# Selected Model: Logistic Regression
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "bank-stress-testing-simulator/processed/bank_health_features.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "risk_scoring_results.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("BANK STRESS TESTING SIMULATOR")
print("STAGE 9: RISK SCORING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "bank_id",
    "scenario_id",
    "bank_condition",
    "bank_condition_code",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. DEFINE MODEL FEATURES
# ============================================================

# These are the same 14 features used in the
# selected Logistic Regression model.

feature_cols = [
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

missing_features = [
    col for col in feature_cols
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )


# ============================================================
# 5. PREPARE X
# ============================================================

X = df[feature_cols].copy()

y = df["bank_condition"].copy()


# ============================================================
# 6. BUILD FINAL LOGISTIC REGRESSION MODEL
# ============================================================

print("\nBuilding final Logistic Regression model...")

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        ),
    ]
)


# ============================================================
# 7. TRAIN MODEL ON FULL DATA
# ============================================================

print("Training final model on complete dataset...")

model.fit(X, y)

print("Model training completed.")


# ============================================================
# 8. GET CLASS PROBABILITIES
# ============================================================

print("\nGenerating risk probabilities...")

probabilities = model.predict_proba(X)

class_names = list(model.named_steps["classifier"].classes_)

print("Model classes:")
print(class_names)


# ============================================================
# 9. SAFELY EXTRACT CLASS PROBABILITIES
# ============================================================

# Get probability for each class by class name.
# This avoids assuming a fixed class order.

probability_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Probability_{class_name}"
        for class_name in class_names
    ],
    index=df.index
)


def get_class_probability(class_name):
    column_name = f"Probability_{class_name}"

    if column_name in probability_df.columns:
        return probability_df[column_name]

    return pd.Series(
        0.0,
        index=df.index
    )


prob_healthy = get_class_probability("Healthy")
prob_stressed = get_class_probability("Stressed")
prob_critical = get_class_probability("Critical")


# ============================================================
# 10. PREDICTED CONDITION
# ============================================================

predicted_condition = model.predict(X)


# ============================================================
# 11. CALCULATE RISK SCORE
# ============================================================

"""
Risk scoring logic:

Healthy  = 0 risk points
Stressed = 50 risk points
Critical = 100 risk points

Expected risk score:

Risk Score =
    (Probability_Stressed × 50)
    +
    (Probability_Critical × 100)

This gives a continuous 0–100 score.

Example:

Healthy probability  = 0.90
Stressed probability = 0.08
Critical probability = 0.02

Risk Score =
(0.08 × 50) + (0.02 × 100)
= 4 + 2
= 6

Therefore the bank has a low estimated risk.

"""

risk_score = (
    (prob_stressed * 50)
    +
    (prob_critical * 100)
)


# Make sure score stays between 0 and 100
risk_score = risk_score.clip(
    lower=0,
    upper=100
)


# ============================================================
# 12. RISK CATEGORY
# ============================================================

def assign_risk_category(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Critical"


risk_category = risk_score.apply(
    assign_risk_category
)


# ============================================================
# 13. PREDICTION CONFIDENCE
# ============================================================

prediction_confidence = probabilities.max(axis=1) * 100


# ============================================================
# 14. BUILD MAIN RISK-SCORING OUTPUT
# ============================================================

risk_results = pd.DataFrame({
    "bank_id": df["bank_id"],
    "scenario_id": df["scenario_id"],

    "actual_condition": df["bank_condition"],

    "predicted_condition": predicted_condition,

    "probability_healthy": prob_healthy,
    "probability_stressed": prob_stressed,
    "probability_critical": prob_critical,

    "risk_score": risk_score,
    "risk_category": risk_category,

    "prediction_confidence": prediction_confidence,
})


# ============================================================
# 15. ROUND NUMERIC VALUES
# ============================================================

risk_results["probability_healthy"] = (
    risk_results["probability_healthy"] * 100
).round(2)

risk_results["probability_stressed"] = (
    risk_results["probability_stressed"] * 100
).round(2)

risk_results["probability_critical"] = (
    risk_results["probability_critical"] * 100
).round(2)

risk_results["risk_score"] = (
    risk_results["risk_score"]
).round(2)

risk_results["prediction_confidence"] = (
    risk_results["prediction_confidence"]
).round(2)


# ============================================================
# 16. RISK SUMMARY
# ============================================================

risk_summary = (
    risk_results["risk_category"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High", "Critical"],
        fill_value=0
    )
    .reset_index()
)

risk_summary.columns = [
    "Risk Category",
    "Number of Bank-Scenarios"
]

risk_summary["Percentage"] = (
    risk_summary["Number of Bank-Scenarios"]
    / len(risk_results)
    * 100
).round(2)


# ============================================================
# 17. PREDICTED CONDITION SUMMARY
# ============================================================

condition_summary = (
    risk_results["predicted_condition"]
    .value_counts()
    .reindex(
        ["Healthy", "Stressed", "Critical"],
        fill_value=0
    )
    .reset_index()
)

condition_summary.columns = [
    "Predicted Condition",
    "Number of Bank-Scenarios"
]

condition_summary["Percentage"] = (
    condition_summary["Number of Bank-Scenarios"]
    / len(risk_results)
    * 100
).round(2)


# ============================================================
# 18. BANK-LEVEL RISK SUMMARY
# ============================================================

bank_summary = (
    risk_results
    .groupby("bank_id")
    .agg(
        total_scenarios=("scenario_id", "count"),
        average_risk_score=("risk_score", "mean"),
        maximum_risk_score=("risk_score", "max"),
        minimum_risk_score=("risk_score", "min"),
        average_critical_probability=(
            "probability_critical",
            "mean"
        ),
        maximum_critical_probability=(
            "probability_critical",
            "max"
        ),
        critical_scenarios=(
            "risk_category",
            lambda x: (x == "Critical").sum()
        ),
        high_or_critical_scenarios=(
            "risk_category",
            lambda x: x.isin(
                ["High", "Critical"]
            ).sum()
        ),
    )
    .reset_index()
)


# ============================================================
# 19. ROUND BANK SUMMARY
# ============================================================

numeric_bank_columns = [
    "average_risk_score",
    "maximum_risk_score",
    "minimum_risk_score",
    "average_critical_probability",
    "maximum_critical_probability",
]

for col in numeric_bank_columns:
    bank_summary[col] = bank_summary[col].round(2)


# ============================================================
# 20. ASSIGN OVERALL BANK RISK
# ============================================================

bank_summary["overall_risk_category"] = (
    bank_summary["average_risk_score"]
    .apply(assign_risk_category)
)


# ============================================================
# 21. SORT BANKS BY RISK
# ============================================================

bank_summary = bank_summary.sort_values(
    by="average_risk_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 22. TOP HIGH-RISK BANK-SCENARIOS
# ============================================================

top_risk_cases = (
    risk_results
    .sort_values(
        by="risk_score",
        ascending=False
    )
    .head(100)
    .copy()
)


# ============================================================
# 23. MODEL INFORMATION
# ============================================================

model_info = pd.DataFrame({
    "Item": [
        "Model",
        "Model Type",
        "Training Dataset Rows",
        "Number of Features",
        "Missing Value Strategy",
        "Scaling",
        "Class Weight",
        "Random State",
        "Risk Score Range",
        "Healthy Risk Weight",
        "Stressed Risk Weight",
        "Critical Risk Weight",
        "Low Risk Range",
        "Moderate Risk Range",
        "High Risk Range",
        "Critical Risk Range",
    ],

    "Value": [
        "Logistic Regression",
        "Multiclass Classification",
        len(df),
        len(feature_cols),
        "Median Imputation",
        "StandardScaler",
        "Balanced",
        42,
        "0 - 100",
        0,
        50,
        100,
        "0 - <25",
        "25 - <50",
        "50 - <75",
        "75 - 100",
    ]
})


# ============================================================
# 24. FEATURE LIST
# ============================================================

feature_information = pd.DataFrame({
    "Feature": feature_cols,
    "Used in Final Model": "Yes"
})


# ============================================================
# 25. SAVE EVERYTHING TO ONE EXCEL FILE
# ============================================================

print("\nSaving risk-scoring results...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    risk_results.to_excel(
        writer,
        sheet_name="Risk Scores",
        index=False
    )

    bank_summary.to_excel(
        writer,
        sheet_name="Bank Summary",
        index=False
    )

    risk_summary.to_excel(
        writer,
        sheet_name="Risk Summary",
        index=False
    )

    condition_summary.to_excel(
        writer,
        sheet_name="Condition Summary",
        index=False
    )

    top_risk_cases.to_excel(
        writer,
        sheet_name="Top Risk Cases",
        index=False
    )

    model_info.to_excel(
        writer,
        sheet_name="Model Info",
        index=False
    )

    feature_information.to_excel(
        writer,
        sheet_name="Features",
        index=False
    )


# ============================================================
# 26. FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK SCORING COMPLETED")
print("=" * 70)

print(f"\nTotal bank-scenario records: {len(risk_results)}")

print("\nRisk Category Distribution:")
print(risk_summary.to_string(index=False))

print("\nPredicted Condition Distribution:")
print(condition_summary.to_string(index=False))

print("\nTop 10 Highest-Risk Bank-Scenarios:")

print(
    risk_results[
        [
            "bank_id",
            "scenario_id",
            "predicted_condition",
            "probability_critical",
            "risk_score",
            "risk_category",
        ]
    ]
    .sort_values(
        by="risk_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\nHighest Average-Risk Banks:")

print(
    bank_summary[
        [
            "bank_id",
            "average_risk_score",
            "maximum_risk_score",
            "critical_scenarios",
            "overall_risk_category",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nExcel sheets created:")
print("1. Risk Scores")
print("2. Bank Summary")
print("3. Risk Summary")
print("4. Condition Summary")
print("5. Top Risk Cases")
print("6. Model Info")
print("7. Features")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)