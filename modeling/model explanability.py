# ============================================================
# MODEL EXPLAINABILITY
# Final Selected Model: Logistic Regression
# Bank Stress Testing Simulator
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "bank-stress-testing-simulator/processed/bank_health_features.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_explainability_results.xlsx"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("MODEL EXPLAINABILITY")
print("Final Model: Logistic Regression")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. DEFINE TARGET AND FEATURES
# ============================================================

TARGET = "bank_condition"

DROP_COLUMNS = [
    "bank_id",
    "scenario_id",
    "bank_condition",
    "bank_condition_code"
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

groups = df["scenario_id"]

print("\nNumber of model features:", X.shape[1])

print("\nFeatures:")
for feature in X.columns:
    print("-", feature)


# ============================================================
# 4. SCENARIO-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()


print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

overlap = set(
    groups_train.unique()
).intersection(
    set(groups_test.unique())
)

print("Scenario overlap:", overlap)


# ============================================================
# 5. FINAL LOGISTIC REGRESSION MODEL
# ============================================================
# These are the same settings used in the selected
# baseline Logistic Regression model.

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining final Logistic Regression model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 7. GET TRAINED MODEL
# ============================================================

model = pipeline.named_steps["model"]

feature_names = X.columns.tolist()

class_names = model.classes_.tolist()

print("\nModel classes:")
print(class_names)


# ============================================================
# 8. EXTRACT COEFFICIENTS
# ============================================================
#
# Logistic Regression is multiclass.
#
# Each class gets a coefficient for every feature.
#
# Positive coefficient:
# feature pushes prediction toward that class.
#
# Negative coefficient:
# feature pushes prediction away from that class.
#
# Since features were standardized, coefficient magnitudes
# can be compared more meaningfully across features.

coefficients = model.coef_


# ============================================================
# 9. COEFFICIENT TABLE
# ============================================================

coefficient_df = pd.DataFrame(
    coefficients.T,
    index=feature_names,
    columns=[
        f"Coefficient - {class_name}"
        for class_name in class_names
    ]
)

coefficient_df = coefficient_df.reset_index()

coefficient_df = coefficient_df.rename(
    columns={"index": "Feature"}
)


# ============================================================
# 10. ABSOLUTE COEFFICIENT IMPORTANCE
# ============================================================
#
# For each feature:
#
# larger absolute coefficient = stronger influence
#
# We calculate the mean absolute coefficient across
# all three classes.

abs_coefficient_matrix = np.abs(coefficients)

mean_abs_coefficient = (
    abs_coefficient_matrix.mean(axis=0)
)

max_abs_coefficient = (
    abs_coefficient_matrix.max(axis=0)
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute Coefficient": mean_abs_coefficient,
    "Maximum Absolute Coefficient": max_abs_coefficient
})


importance_df = importance_df.sort_values(
    "Mean Absolute Coefficient",
    ascending=False
).reset_index(drop=True)


importance_df[
    "Mean Absolute Coefficient"
] = importance_df[
    "Mean Absolute Coefficient"
].round(6)


importance_df[
    "Maximum Absolute Coefficient"
] = importance_df[
    "Maximum Absolute Coefficient"
].round(6)


# ============================================================
# 11. CLASS-SPECIFIC FEATURE IMPORTANCE
# ============================================================

class_importance_tables = {}

for i, class_name in enumerate(class_names):

    temp = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients[i],
        "Absolute Coefficient": np.abs(
            coefficients[i]
        )
    })

    temp = temp.sort_values(
        "Absolute Coefficient",
        ascending=False
    ).reset_index(drop=True)

    temp[
        "Coefficient"
    ] = temp[
        "Coefficient"
    ].round(6)

    temp[
        "Absolute Coefficient"
    ] = temp[
        "Absolute Coefficient"
    ].round(6)

    class_importance_tables[class_name] = temp


# ============================================================
# 12. TOP FEATURES FOR EACH CLASS
# ============================================================

top_features_rows = []

TOP_N = 10

for class_name in class_names:

    temp = class_importance_tables[class_name]

    top_temp = temp.head(TOP_N)

    for rank, (_, row) in enumerate(
        top_temp.iterrows(),
        start=1
    ):

        top_features_rows.append({
            "Class": class_name,
            "Rank": rank,
            "Feature": row["Feature"],
            "Coefficient": row["Coefficient"],
            "Absolute Coefficient": row[
                "Absolute Coefficient"
            ]
        })


top_features_df = pd.DataFrame(
    top_features_rows
)


# ============================================================
# 13. POSITIVE / NEGATIVE DRIVERS
# ============================================================
#
# Positive coefficient = pushes prediction toward class
#
# Negative coefficient = pushes prediction away from class

driver_rows = []

for i, class_name in enumerate(class_names):

    temp = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients[i]
    })

    positive = temp.sort_values(
        "Coefficient",
        ascending=False
    ).head(10)

    negative = temp.sort_values(
        "Coefficient",
        ascending=True
    ).head(10)

    for rank, (_, row) in enumerate(
        positive.iterrows(),
        start=1
    ):

        driver_rows.append({
            "Class": class_name,
            "Direction": "Positive Driver",
            "Rank": rank,
            "Feature": row["Feature"],
            "Coefficient": round(
                row["Coefficient"],
                6
            )
        })

    for rank, (_, row) in enumerate(
        negative.iterrows(),
        start=1
    ):

        driver_rows.append({
            "Class": class_name,
            "Direction": "Negative Driver",
            "Rank": rank,
            "Feature": row["Feature"],
            "Coefficient": round(
                row["Coefficient"],
                6
            )
        })


drivers_df = pd.DataFrame(
    driver_rows
)


# ============================================================
# 14. TEST-SET PREDICTIONS
# ============================================================

test_predictions = pipeline.predict(
    X_test
)

test_probabilities = pipeline.predict_proba(
    X_test
)


# ============================================================
# 15. EXPLANATION FOR TEST RECORDS
# ============================================================
#
# For each test record, identify:
# - predicted class
# - highest probability
# - strongest feature contribution toward
#   the predicted class
#
# Contribution is calculated as:
#
# standardized feature value × class coefficient

X_test_imputed = pipeline.named_steps[
    "imputer"
].transform(X_test)

X_test_scaled = pipeline.named_steps[
    "scaler"
].transform(X_test_imputed)


local_explanation_rows = []

for row_position in range(
    len(X_test_scaled)
):

    predicted_class = test_predictions[
        row_position
    ]

    predicted_class_index = list(
        class_names
    ).index(predicted_class)

    row_values = X_test_scaled[
        row_position
    ]

    class_coefficients = coefficients[
        predicted_class_index
    ]

    contributions = (
        row_values * class_coefficients
    )

    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    contribution_df["Absolute Contribution"] = (
        contribution_df["Contribution"].abs()
    )

    strongest = contribution_df.sort_values(
        "Absolute Contribution",
        ascending=False
    ).iloc[0]

    probability = test_probabilities[
        row_position,
        predicted_class_index
    ]

    local_explanation_rows.append({
        "Bank ID": df.iloc[
            test_idx[row_position]
        ]["bank_id"],

        "Scenario ID": df.iloc[
            test_idx[row_position]
        ]["scenario_id"],

        "Actual Condition": y_test.iloc[
            row_position
        ],

        "Predicted Condition": predicted_class,

        "Prediction Probability": round(
            probability,
            6
        ),

        "Strongest Feature": strongest[
            "Feature"
        ],

        "Strongest Feature Contribution": round(
            strongest["Contribution"],
            6
        )
    })


local_explanation_df = pd.DataFrame(
    local_explanation_rows
)


# ============================================================
# 16. ROUND COEFFICIENT TABLE
# ============================================================

for column in coefficient_df.columns:

    if column != "Feature":

        coefficient_df[column] = (
            coefficient_df[column]
            .round(6)
        )


# ============================================================
# 17. MODEL INFORMATION
# ============================================================

model_info_df = pd.DataFrame({
    "Item": [
        "Final Model",
        "Target",
        "Number of Features",
        "Training Rows",
        "Testing Rows",
        "Missing Value Handling",
        "Feature Scaling",
        "Class Weight",
        "Scenario-Aware Split",
        "Random State"
    ],

    "Value": [
        "Logistic Regression",
        TARGET,
        X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
        "Median Imputation",
        "StandardScaler",
        "Balanced",
        "Yes",
        42
    ]
})


# ============================================================
# 18. SAVE ONE EXCEL FILE
# ============================================================

print("\nSaving explainability results...")

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl"
) as writer:

    # Sheet 1
    model_info_df.to_excel(
        writer,
        sheet_name="Model Info",
        index=False
    )

    # Sheet 2
    coefficient_df.to_excel(
        writer,
        sheet_name="Coefficients",
        index=False
    )

    # Sheet 3
    importance_df.to_excel(
        writer,
        sheet_name="Overall Importance",
        index=False
    )

    # Sheet 4
    top_features_df.to_excel(
        writer,
        sheet_name="Top Features",
        index=False
    )

    # Sheet 5
    drivers_df.to_excel(
        writer,
        sheet_name="Feature Drivers",
        index=False
    )

    # Sheet 6
    local_explanation_df.to_excel(
        writer,
        sheet_name="Test Explanations",
        index=False
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MODEL EXPLAINABILITY COMPLETED")
print("=" * 65)

print("\nFinal Model:")
print("Logistic Regression")

print("\nTop 10 Overall Features:")

print(
    importance_df.head(10).to_string(
        index=False
    )
)

print("\nOutput file:")
print(OUTPUT_PATH)

print("\n Excel file was created.")

print("\nExcel sheets:")
print("1. Model Info")
print("2. Coefficients")
print("3. Overall Importance")
print("4. Top Features")
print("5. Feature Drivers")
print("6. Test Explanations")