# ============================================================
# BANK STRESS TESTING SIMULATOR
# INVESTIGATION OUTPUT
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "outputs/risk_scoring_results.xlsx"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "investigation_output.xlsx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)


# ============================================================
# 2. LOAD RISK-SCORING RESULTS
# ============================================================

print("=" * 70)
print("BANK STRESS TESTING SIMULATOR")
print("STAGE 10: INVESTIGATION OUTPUT")
print("=" * 70)

print("\nLoading risk-scoring results...")

risk_df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Risk Scores"
)

print(f"Risk records loaded: {risk_df.shape}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "bank_id",
    "scenario_id",
    "actual_condition",
    "predicted_condition",
    "probability_healthy",
    "probability_stressed",
    "probability_critical",
    "risk_score",
    "risk_category",
    "prediction_confidence",
]

missing_columns = [
    col
    for col in required_columns
    if col not in risk_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. DATA CLEANING / TYPES
# ============================================================

numeric_columns = [
    "probability_healthy",
    "probability_stressed",
    "probability_critical",
    "risk_score",
    "prediction_confidence",
]

for col in numeric_columns:
    risk_df[col] = pd.to_numeric(
        risk_df[col],
        errors="coerce"
    )


# ============================================================
# 5. INVESTIGATION PRIORITY
# ============================================================

"""
Investigation priority is based primarily on risk category.

Critical -> Immediate
High     -> High
Moderate -> Medium
Low      -> Routine
"""

priority_map = {
    "Critical": "Immediate",
    "High": "High",
    "Moderate": "Medium",
    "Low": "Routine",
}

risk_df["investigation_priority"] = (
    risk_df["risk_category"]
    .map(priority_map)
)


# ============================================================
# 6. INVESTIGATION STATUS
# ============================================================

"""
This is a simulated workflow field.

It does NOT claim that a real compliance
investigation has occurred.

All cases initially start as "Pending Review".
"""

risk_df["investigation_status"] = "Pending Review"


# ============================================================
# 7. PRIMARY RISK DRIVER
# ============================================================

def identify_primary_driver(row):

    probabilities = {
        "Critical Risk Probability": row[
            "probability_critical"
        ],

        "Stressed Risk Probability": row[
            "probability_stressed"
        ],

        "Healthy Probability": row[
            "probability_healthy"
        ],
    }

    strongest = max(
        probabilities,
        key=probabilities.get
    )

    return strongest


risk_df["primary_risk_signal"] = (
    risk_df.apply(
        identify_primary_driver,
        axis=1
    )
)


# ============================================================
# 8. INVESTIGATION REASON
# ============================================================

def generate_investigation_reason(row):

    category = row["risk_category"]

    critical_prob = row["probability_critical"]

    predicted = row["predicted_condition"]

    if category == "Critical":
        return (
            f"Critical risk score ({row['risk_score']:.2f}) "
            f"with {critical_prob:.2f}% probability of Critical condition."
        )

    elif category == "High":
        return (
            f"High risk score ({row['risk_score']:.2f}) "
            f"requires enhanced review."
        )

    elif category == "Moderate":
        return (
            f"Moderate risk score ({row['risk_score']:.2f}); "
            f"continued monitoring recommended."
        )

    else:
        return (
            f"Low risk score ({row['risk_score']:.2f}); "
            f"routine monitoring."
        )


risk_df["investigation_reason"] = (
    risk_df.apply(
        generate_investigation_reason,
        axis=1
    )
)


# ============================================================
# 9. ACTUAL VS PREDICTED STATUS
# ============================================================

def prediction_status(row):

    if row["actual_condition"] == row["predicted_condition"]:
        return "Correct"

    return "Mismatch"


risk_df["prediction_status"] = (
    risk_df.apply(
        prediction_status,
        axis=1
    )
)


# ============================================================
# 10. CRITICAL ALERT FLAG
# ============================================================

risk_df["critical_alert"] = np.where(
    (
        (risk_df["risk_category"] == "Critical")
        |
        (risk_df["probability_critical"] >= 75)
    ),
    "ALERT",
    "Normal"
)


# ============================================================
# 11. HIGH-RISK INVESTIGATION QUEUE
# ============================================================

investigation_queue = (
    risk_df[
        risk_df["risk_category"].isin(
            ["Critical", "High"]
        )
    ]
    .copy()
)


investigation_queue = investigation_queue.sort_values(
    by=[
        "risk_score",
        "probability_critical",
        "prediction_confidence",
    ],
    ascending=[
        False,
        False,
        False,
    ]
)


investigation_queue = investigation_queue.reset_index(
    drop=True
)


investigation_queue.insert(
    0,
    "investigation_rank",
    range(
        1,
        len(investigation_queue) + 1
    )
)


# ============================================================
# 12. CRITICAL CASES ONLY
# ============================================================

critical_cases = (
    risk_df[
        risk_df["risk_category"] == "Critical"
    ]
    .copy()
)


critical_cases = critical_cases.sort_values(
    by="risk_score",
    ascending=False
).reset_index(drop=True)


critical_cases.insert(
    0,
    "critical_rank",
    range(
        1,
        len(critical_cases) + 1
    )
)


# ============================================================
# 13. MODEL PREDICTION MISMATCHES
# ============================================================

prediction_mismatches = (
    risk_df[
        risk_df["prediction_status"] == "Mismatch"
    ]
    .copy()
)


prediction_mismatches = prediction_mismatches.sort_values(
    by="risk_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 14. BANK-LEVEL INVESTIGATION SUMMARY
# ============================================================

bank_summary = (
    risk_df
    .groupby("bank_id")
    .agg(
        total_scenarios=(
            "scenario_id",
            "count"
        ),

        average_risk_score=(
            "risk_score",
            "mean"
        ),

        maximum_risk_score=(
            "risk_score",
            "max"
        ),

        average_critical_probability=(
            "probability_critical",
            "mean"
        ),

        maximum_critical_probability=(
            "probability_critical",
            "max"
        ),

        critical_cases=(
            "risk_category",
            lambda x: (
                x == "Critical"
            ).sum()
        ),

        high_risk_cases=(
            "risk_category",
            lambda x: (
                x == "High"
            ).sum()
        ),

        high_or_critical_cases=(
            "risk_category",
            lambda x: x.isin(
                [
                    "High",
                    "Critical"
                ]
            ).sum()
        ),

        prediction_mismatches=(
            "prediction_status",
            lambda x: (
                x == "Mismatch"
            ).sum()
        ),
    )
    .reset_index()
)


# ============================================================
# 15. BANK RISK CATEGORY
# ============================================================

def bank_risk_category(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Critical"


bank_summary[
    "overall_risk_category"
] = (
    bank_summary[
        "average_risk_score"
    ]
    .apply(bank_risk_category)
)


# ============================================================
# 16. BANK INVESTIGATION PRIORITY
# ============================================================

def bank_priority(row):

    if row["critical_cases"] > 0:
        return "Immediate"

    elif row["high_risk_cases"] > 0:
        return "High"

    elif row["average_risk_score"] >= 25:
        return "Medium"

    else:
        return "Routine"


bank_summary[
    "investigation_priority"
] = (
    bank_summary.apply(
        bank_priority,
        axis=1
    )
)


# ============================================================
# 17. ROUND BANK SUMMARY
# ============================================================

round_columns = [
    "average_risk_score",
    "maximum_risk_score",
    "average_critical_probability",
    "maximum_critical_probability",
]

for col in round_columns:
    bank_summary[col] = (
        bank_summary[col]
        .round(2)
    )


# ============================================================
# 18. SORT BANK SUMMARY
# ============================================================

bank_summary = bank_summary.sort_values(
    by=[
        "average_risk_score",
        "critical_cases",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(drop=True)


# ============================================================
# 19. SCENARIO-LEVEL SUMMARY
# ============================================================

scenario_summary = (
    risk_df
    .groupby("scenario_id")
    .agg(
        total_banks=(
            "bank_id",
            "count"
        ),

        average_risk_score=(
            "risk_score",
            "mean"
        ),

        maximum_risk_score=(
            "risk_score",
            "max"
        ),

        critical_cases=(
            "risk_category",
            lambda x: (
                x == "Critical"
            ).sum()
        ),

        high_risk_cases=(
            "risk_category",
            lambda x: (
                x == "High"
            ).sum()
        ),

        average_critical_probability=(
            "probability_critical",
            "mean"
        ),
    )
    .reset_index()
)


scenario_summary[
    "critical_case_percentage"
] = (
    scenario_summary[
        "critical_cases"
    ]
    /
    scenario_summary[
        "total_banks"
    ]
    * 100
)


scenario_summary[
    "high_or_critical_percentage"
] = (
    (
        scenario_summary[
            "critical_cases"
        ]
        +
        scenario_summary[
            "high_risk_cases"
        ]
    )
    /
    scenario_summary[
        "total_banks"
    ]
    * 100
)


scenario_summary[
    "average_risk_score"
] = (
    scenario_summary[
        "average_risk_score"
    ].round(2)
)

scenario_summary[
    "maximum_risk_score"
] = (
    scenario_summary[
        "maximum_risk_score"
    ].round(2)
)

scenario_summary[
    "average_critical_probability"
] = (
    scenario_summary[
        "average_critical_probability"
    ].round(2)
)

scenario_summary[
    "critical_case_percentage"
] = (
    scenario_summary[
        "critical_case_percentage"
    ].round(2)
)

scenario_summary[
    "high_or_critical_percentage"
] = (
    scenario_summary[
        "high_or_critical_percentage"
    ].round(2)
)


scenario_summary = scenario_summary.sort_values(
    by="average_risk_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 20. OVERALL DASHBOARD KPIs
# ============================================================

total_records = len(risk_df)

total_banks = (
    risk_df["bank_id"]
    .nunique()
)

total_scenarios = (
    risk_df["scenario_id"]
    .nunique()
)

critical_records = (
    risk_df["risk_category"]
    .eq("Critical")
    .sum()
)

high_records = (
    risk_df["risk_category"]
    .eq("High")
    .sum()
)

high_or_critical_records = (
    risk_df["risk_category"]
    .isin(
        [
            "High",
            "Critical"
        ]
    )
    .sum()
)

average_risk = (
    risk_df["risk_score"]
    .mean()
)

maximum_risk = (
    risk_df["risk_score"]
    .max()
)

mismatch_count = (
    risk_df["prediction_status"]
    .eq("Mismatch")
    .sum()
)


dashboard_kpis = pd.DataFrame({
    "KPI": [
        "Total Bank-Scenario Records",
        "Total Banks",
        "Total Scenarios",
        "Critical Cases",
        "High-Risk Cases",
        "High + Critical Cases",
        "Average Risk Score",
        "Maximum Risk Score",
        "Prediction Mismatches",
    ],

    "Value": [
        total_records,
        total_banks,
        total_scenarios,
        critical_records,
        high_records,
        high_or_critical_records,
        round(average_risk, 2),
        round(maximum_risk, 2),
        mismatch_count,
    ]
})


# ============================================================
# 21. RISK CATEGORY DISTRIBUTION
# ============================================================

category_distribution = (
    risk_df["risk_category"]
    .value_counts()
    .reindex(
        [
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        fill_value=0
    )
    .reset_index()
)


category_distribution.columns = [
    "Risk Category",
    "Number of Cases",
]


category_distribution[
    "Percentage"
] = (
    category_distribution[
        "Number of Cases"
    ]
    /
    total_records
    * 100
).round(2)


# ============================================================
# 22. CONDITION DISTRIBUTION
# ============================================================

condition_distribution = (
    risk_df["predicted_condition"]
    .value_counts()
    .reindex(
        [
            "Healthy",
            "Stressed",
            "Critical",
        ],
        fill_value=0
    )
    .reset_index()
)


condition_distribution.columns = [
    "Predicted Condition",
    "Number of Cases",
]


condition_distribution[
    "Percentage"
] = (
    condition_distribution[
        "Number of Cases"
    ]
    /
    total_records
    * 100
).round(2)


# ============================================================
# 23. INVESTIGATION OUTPUT COLUMNS
# ============================================================

investigation_columns = [
    "bank_id",
    "scenario_id",
    "actual_condition",
    "predicted_condition",
    "prediction_status",
    "probability_healthy",
    "probability_stressed",
    "probability_critical",
    "risk_score",
    "risk_category",
    "investigation_priority",
    "critical_alert",
    "prediction_confidence",
    "primary_risk_signal",
    "investigation_reason",
    "investigation_status",
]


investigation_output = risk_df[
    investigation_columns
].copy()


investigation_output = investigation_output.sort_values(
    by=[
        "risk_score",
        "probability_critical",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(drop=True)


# ============================================================
# 24. SAVE EXCEL
# ============================================================

print("\nSaving investigation output...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    dashboard_kpis.to_excel(
        writer,
        sheet_name="Dashboard KPIs",
        index=False
    )

    investigation_output.to_excel(
        writer,
        sheet_name="Investigation Output",
        index=False
    )

    investigation_queue.to_excel(
        writer,
        sheet_name="Investigation Queue",
        index=False
    )

    critical_cases.to_excel(
        writer,
        sheet_name="Critical Cases",
        index=False
    )

    prediction_mismatches.to_excel(
        writer,
        sheet_name="Prediction Mismatches",
        index=False
    )

    bank_summary.to_excel(
        writer,
        sheet_name="Bank Summary",
        index=False
    )

    scenario_summary.to_excel(
        writer,
        sheet_name="Scenario Summary",
        index=False
    )

    category_distribution.to_excel(
        writer,
        sheet_name="Risk Distribution",
        index=False
    )

    condition_distribution.to_excel(
        writer,
        sheet_name="Condition Distribution",
        index=False
    )


# ============================================================
# 25. CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("INVESTIGATION OUTPUT COMPLETED")
print("=" * 70)

print("\nDashboard KPIs:")
print(dashboard_kpis.to_string(index=False))

print("\nRisk Distribution:")
print(
    category_distribution.to_string(
        index=False
    )
)

print("\nTop 10 Investigation Cases:")

print(
    investigation_queue[
        [
            "investigation_rank",
            "bank_id",
            "scenario_id",
            "predicted_condition",
            "probability_critical",
            "risk_score",
            "risk_category",
            "investigation_priority",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTop 10 Highest-Risk Banks:")

print(
    bank_summary[
        [
            "bank_id",
            "average_risk_score",
            "maximum_risk_score",
            "critical_cases",
            "high_risk_cases",
            "overall_risk_category",
            "investigation_priority",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nHighest-Risk Scenarios:")

print(
    scenario_summary[
        [
            "scenario_id",
            "average_risk_score",
            "maximum_risk_score",
            "critical_cases",
            "high_risk_cases",
            "critical_case_percentage",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nExcel sheets created:")
print("1. Dashboard KPIs")
print("2. Investigation Output")
print("3. Investigation Queue")
print("4. Critical Cases")
print("5. Prediction Mismatches")
print("6. Bank Summary")
print("7. Scenario Summary")
print("8. Risk Distribution")
print("9. Condition Distribution")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)