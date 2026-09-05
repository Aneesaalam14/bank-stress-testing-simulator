# ============================================================
# MODEL SELECTION
# Bank Stress Testing Simulator
# ============================================================

import os
import pandas as pd


# ============================================================
# 1. OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. EXISTING BASELINE MODEL RESULTS
# ============================================================
# These are the results obtained from the three baseline models.

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        0.8605,
        0.8415,
        0.8785
    ],

    "Balanced Accuracy": [
        0.8610,
        0.8230,
        0.8604
    ],

    "Macro F1": [
        0.8551,
        0.8245,
        0.8649
    ],

    "Critical Recall": [
        0.8782,
        0.7890,
        0.8232
    ]
})


# ============================================================
# 3. IDENTIFY BEST MODEL FOR EACH METRIC
# ============================================================

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Critical Recall"
]

best_model_rows = []

for metric in metric_columns:

    best_index = results[metric].idxmax()

    best_model_rows.append({
        "Metric": metric,
        "Best Model": results.loc[best_index, "Model"],
        "Score": results.loc[best_index, metric]
    })

best_by_metric = pd.DataFrame(best_model_rows)


# ============================================================
# 4. MODEL SELECTION CRITERIA
# ============================================================
# For this project:
#
# Macro F1 = primary overall classification metric
# Balanced Accuracy = important because classes are imbalanced
# Critical Recall = important because missing a Critical bank
#                  is more serious
#
# We therefore use:
#
# 40% Macro F1
# 35% Balanced Accuracy
# 25% Critical Recall
#
# Accuracy is reported but not used as the main selection score.


results["Selection Score"] = (
    0.40 * results["Macro F1"]
    + 0.35 * results["Balanced Accuracy"]
    + 0.25 * results["Critical Recall"]
)


# ============================================================
# 5. RANK MODELS
# ============================================================

results["Rank"] = (
    results["Selection Score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


results = results.sort_values(
    "Rank"
).reset_index(drop=True)


# ============================================================
# 6. FINAL MODEL
# ============================================================

final_model = results.loc[0, "Model"]
final_score = results.loc[0, "Selection Score"]


# ============================================================
# 7. MODEL SELECTION SUMMARY
# ============================================================

selection_summary = pd.DataFrame({
    "Item": [
        "Final Selected Model",
        "Selection Score",
        "Primary Metric",
        "Selection Method",
        "Number of Models Compared"
    ],

    "Value": [
        final_model,
        round(final_score, 4),
        "Macro F1",
        "Weighted combination of Macro F1, Balanced Accuracy and Critical Recall",
        len(results)
    ]
})


# ============================================================
# 8. FORMAT RESULTS FOR EXCEL
# ============================================================

results_excel = results.copy()

for metric in metric_columns:
    results_excel[metric] = (
        results_excel[metric] * 100
    ).round(2)

results_excel["Selection Score"] = (
    results_excel["Selection Score"] * 100
).round(2)


best_by_metric_excel = best_by_metric.copy()

best_by_metric_excel["Score"] = (
    best_by_metric_excel["Score"] * 100
).round(2)


# ============================================================
# 9. SAVE ONE EXCEL FILE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "model_selection_results.xlsx"
)


with pd.ExcelWriter(
    output_path,
    engine="openpyxl"
) as writer:

    # Sheet 1
    results_excel.to_excel(
        writer,
        sheet_name="Model Ranking",
        index=False
    )

    # Sheet 2
    best_by_metric_excel.to_excel(
        writer,
        sheet_name="Best By Metric",
        index=False
    )

    # Sheet 3
    selection_summary.to_excel(
        writer,
        sheet_name="Final Selection",
        index=False
    )


# ============================================================
# 10. FINAL TERMINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("MODEL SELECTION COMPLETED")
print("=" * 65)

print("\nModel Ranking:")

print(
    results_excel[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Critical Recall",
            "Selection Score"
        ]
    ].to_string(index=False)
)

print("\n" + "-" * 65)

print(
    f"FINAL SELECTED MODEL: {final_model}"
)

print(
    f"FINAL SELECTION SCORE: {final_score * 100:.2f}%"
)

print("-" * 65)

print("\nBest model by individual metric:")

for _, row in best_by_metric_excel.iterrows():

    print(
        f"{row['Metric']}: "
        f"{row['Best Model']} "
        f"({row['Score']:.2f}%)"
    )

print("\nOutput file:")
print(output_path)

print("\nExcel file was created.")