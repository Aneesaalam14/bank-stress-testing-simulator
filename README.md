# Bank Stress-Testing Simulator

A data-driven simulator that answers one question for a Pakistani bank facing a macroeconomic shock: will it stay Healthy, become Stressed, or turn Critical.

## Overview

Pakistani banks are exposed to macroeconomic shocks such as inflation spikes, interest rate hikes, currency devaluation, and GDP contractions, but there is no accessible, data-driven way to see how a given shock would actually affect a bank's capital adequacy (CAR), asset quality (NPL), liquidity, and profitability. Regulators and analysts are left with two options: a manual, bank-by-bank stress test, which is slow and inconsistent across analysts, or no tool at all for quickly asking "what happens to bank health if X shock hits."

This project builds a system to answer that question directly. It learns sector-level default risk from real loan data, applies that risk to 40 synthetic but realistic Pakistani banks, tests each bank against 500 macroeconomic shock scenarios, and produces a final health label, Healthy, Stressed, or Critical, for every bank and scenario combination.

## Contents / structure

```
bank-stress-testing-simulator-main/
├── README.md
├── notebooks/
│   ├── data_cleaning.ipynb                   # cleans all 5 
│   ├── eda_bank_profiles.ipynb              
│   ├── eda_bank_stress_panel.ipynb           
│   ├── eda_loan_portfolio.ipynb              
│   ├── eda_macro_scenarios.ipynb            
│   └── eda_macro_stress_scenarios.ipynb      
├── processed/                                # cleaned data, 
│   ├── bank_profiles_clean.csv
│   ├── bank_stress_simulated_panel_clean.csv
│   ├── loan_portfolio_clean.csv
│   ├── macro_scenarios_clean.csv
│   └── macro_stress_scenarios_clean.csv
├── raw/                                      # original
│   ├── bank_profiles.csv
│   ├── bank_stress_simulated_panel.csv
│   ├── loan_portfolio.csv
│   ├── macro_scenarios.csv
│   └── macro_stress_scenarios.csv
└── Schema/                                   
    ├── EDA_bank_profile.sql
    ├── EDA_bank_stress_simulated.sql
    ├── EDA_Loan_portfolio.sql
    ├── EDA_macro_scenerios.sql
    ├── EDA_macro_stress_scenarios.sql
    ├── SCHEMA.sql
    └── Script.sql
```


## Models

`models/` holds the trained model artifacts. These are generated locally by the `modeling/` notebooks and are not version-controlled: `random_forest_model.pkl` is roughly 145 MB, which exceeds GitHub's 100 MB file-size limit, so it is gitignored (see `.gitignore`). To regenerate it, run `modeling/02_random_forest.ipynb`, which trains the random forest pipeline and saves it to `models/random_forest_model.pkl` with `joblib.dump`.


## Data description

| File | Grain | Rows (cleaned) | Key columns | Notes |
|---|---|---|---|---|
| `bank_profiles.csv` | one row per bank | 40 | `bank_id`, `size_tier`, `total_assets_usd`, `total_loans_usd`, `deposit_base_usd`, `baseline_car_pct`, `baseline_liquidity_ratio_pct`, `baseline_roa_pct`, `sector_concentration`, `bank_risk_factor`, ten `sector_wt_*` columns | Sector weights sum to roughly 1.0 per bank |
| `loan_portfolio.csv` | one row per loan | ~3,000 | `loan_id`, `sector`, `pd_annual`, `lgd`, `ead`, `rwa`, `loan_amount` | Loan-level basis the sector risk profiles are learned from |
| `macro_scenarios.csv` | one row per scenario | 500 | `scenario_id`, `stress_intensity`, `scenario_severity`, `gdp_shock_pp`, `unemp_shock_pp`, `rate_shock_pp`, `credit_spread_bps`, `inflation_shock_pp`, `fx_devaluation_pct` | `scenario_severity` runs baseline to severe |
| `macro_stress_scenarios.csv` | one row per scenario x sector | 60 (6 scenarios x 10 sectors) | `scenario`, `sector`, `pd_multiplier`, `base_lgd`, `stressed_lgd`, plus the same shock columns as above | Includes `gfc_like` and `covid_like`, which are not in `macro_scenarios.csv` and are not tested in the main run, see caveat below |
| `bank_stress_simulated_panel.csv` | one row per bank x scenario | ~20,000 (40 banks x ~500 scenarios) | `bank_id`, `scenario_id`, shock inputs, `projected_npl_ratio_pct`, `car_after_pct`, `liquidity_after_pct`, `roa_after_pct`, `bank_condition` | Main output of the simulation; `bank_condition` is Healthy, Stressed, or Critical |

**Known caveats**

- The `gfc_like` / `covid_like` gap described above. Results in the panel do not cover those named historical scenarios.
- A handful of rows across the panel and the loan portfolio had to be corrected rather than simply dropped, most notably a decimal-point-shift error in `roa_after_pct` and a `-1` sentinel value standing in for missing data in `liquidity_after_pct`. See `eda_bank_stress_panel.ipynb` for the detail.
- All data here is synthetic. It is built to resemble real-world Pakistani bank data and deliberately includes realistic data-quality issues, but it is not drawn from actual bank filings.

## Methodology / approach

1. **EDA.** Each raw file gets its own notebook that inspects structure, quantifies every data-quality issue found, and plots the main distributions and relationships, with a short conclusion after each step.
2. **Cleaning.** Each raw file had its own mix of problems: inconsistent category spelling, numbers stored as text with currency symbols, percent signs, or thousand separators, missing values, a few sentinel values standing in for missing data, exact duplicate rows, and one identified unit-scale error. `data_cleaning.ipynb` fixes all of these and writes a clean version of each file to `processed/`, without altering the raw files.



## Results / key findings

- Across all 40 banks and roughly 500 scenarios, banks end up Healthy in about 52% of bank-scenario pairs, Stressed in about 33%, and Critical in about 16%.
- The Healthy share falls steadily as scenario severity increases: about 74% Healthy at baseline versus about 30% Healthy under severe scenarios, with Critical outcomes rising from about 8% to about 26% over the same range. This is the expected direction for a stress test and is a useful sanity check that the underlying simulation behaves sensibly.
- Loan-level PD and the macro stress file's PD multipliers agree with each other on which sectors are riskier: cyclical sectors such as energy and real estate carry higher risk under stress than steadier sectors such as utilities and consumer.
- Shock variables move together the way a real macro shock would: GDP shocks are mostly negative, GDP and unemployment shocks move in opposite directions, and credit spreads and rate shocks widen as scenario severity increases.

