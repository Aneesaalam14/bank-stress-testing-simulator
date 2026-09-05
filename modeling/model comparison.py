# ============================================================
# MODEL COMPARISON
# Bank Stress Testing Simulator
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
import os

# ============================================================
# 1. OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. MODEL RESULTS
# ============================================================

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

# Convert metrics to percentage

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Critical Recall"
]

results_percentage = results.copy()

for col in metric_columns:
    results_percentage[col] = (
        results_percentage[col] * 100
    ).round(2)

# ============================================================
# 3. FIND BEST MODEL FOR EACH METRIC
# ============================================================

best_models = pd.DataFrame({
    "Metric": metric_columns,
    "Best Model": [
        results.loc[results["Accuracy"].idxmax(), "Model"],
        results.loc[results["Balanced Accuracy"].idxmax(), "Model"],
        results.loc[results["Macro F1"].idxmax(), "Model"],
        results.loc[results["Critical Recall"].idxmax(), "Model"]
    ],
    "Best Score": [
        results["Accuracy"].max() * 100,
        results["Balanced Accuracy"].max() * 100,
        results["Macro F1"].max() * 100,
        results["Critical Recall"].max() * 100
    ]
})

best_models["Best Score"] = best_models["Best Score"].round(2)

# ============================================================
# 4. CREATE ONE PNG WITH TWO CHARTS
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)

# ------------------------------------------------------------
# CHART 1: MODEL PERFORMANCE
# ------------------------------------------------------------

performance_metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1"
]

x = range(len(results["Model"]))
width = 0.25

for i, metric in enumerate(performance_metrics):

    axes[0].bar(
        [p + (i - 1) * width for p in x],
        results[metric] * 100,
        width=width,
        label=metric
    )

axes[0].set_title(
    "Model Performance Comparison",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_ylabel("Score (%)")

axes[0].set_xticks(list(x))
axes[0].set_xticklabels(
    results["Model"],
    rotation=15
)

axes[0].set_ylim(0, 100)

axes[0].legend()

axes[0].grid(
    axis="y",
    alpha=0.3
)

# ------------------------------------------------------------
# CHART 2: CRITICAL RECALL
# ------------------------------------------------------------

axes[1].bar(
    results["Model"],
    results["Critical Recall"] * 100
)

axes[1].set_title(
    "Critical Recall Comparison",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_ylabel("Critical Recall (%)")

axes[1].set_ylim(0, 100)

axes[1].tick_params(
    axis="x",
    rotation=15
)

axes[1].grid(
    axis="y",
    alpha=0.3
)

# Add percentage labels

for i, value in enumerate(
    results["Critical Recall"] * 100
):

    axes[1].text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=10
    )

# ============================================================
# 5. FINAL PNG LAYOUT
# ============================================================

fig.suptitle(
    "Bank Stress Testing - Model Comparison",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCombined comparison chart saved:")
print(png_path)

# ============================================================
# 6. CREATE ONE EXCEL FILE
# ============================================================

excel_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_results.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    # Sheet 1
    results_percentage.to_excel(
        writer,
        sheet_name="Model Comparison",
        index=False
    )

    # Sheet 2
    best_models.to_excel(
        writer,
        sheet_name="Best Models",
        index=False
    )

# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("MODEL COMPARISON COMPLETED")
print("==============================================")

print("\nExcel output:")
print(excel_path)

print("\nPNG output:")
print(png_path)

print("\nFiles created:")
print("1. model_comparison_results.xlsx")
print("2. model_comparison.png")

