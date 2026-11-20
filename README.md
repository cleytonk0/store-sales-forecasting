# Store Sales Forecasting — Exploratory Data Analysis and Feature Engineering

## 1. Project Overview

This repository documents the exploratory data analysis (EDA), data preparation,
and temporal feature-engineering stage for Corporación Favorita's Store Sales
time-series challenge. It is a portfolio project about preparing sales data; it
does **not** implement or evaluate a forecasting model.

## 2. Business Problem

Retailers need to understand how product-family sales vary by store and date to
support inventory and operational planning. The original challenge asks
participants to forecast future sales. The target in this project is `sales`,
the recorded sales volume, not product prices or revenue.

## 3. Dataset

The data comes from the [Store Sales - Time Series Forecasting Kaggle
competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
The workflow uses:

- `train.csv` and `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`

Although `transactions.csv` is mentioned in exploratory notes, the notebook
does not load or integrate it into the sales table. It is therefore not a
required input to the scripts.

## 4. Project Objectives

- Inspect dataset structure, types, missing values, and duplicate rows.
- Explore sales, promotions, time patterns, store geography, oil prices, and
  holidays using the analyses already present in the notebook.
- Integrate store, oil, and holiday information with explicit join keys.
- Make daily series complete before calculating row-based lags and rolling
  statistics.
- Keep assumptions and unimplemented forecasting steps explicit.

## 5. Technologies and Libraries

- Python 3.10+
- Pandas
- Matplotlib
- Seaborn
- Jupyter

NumPy is not imported directly by the notebook or scripts. KaggleHub is also
not required: the project reads downloaded CSV files from `data/` and does not
perform authenticated downloads.

## 6. Exploratory Data Analysis

The notebook examines dimensions, data types, missing values, duplicate rows,
category counts, the sales target, promotion patterns, oil-price history, store
geography, and the holiday/event calendar. Its visualizations include oil
prices, store and holiday distributions, promotion prevalence, sales averages
by weekday/weekend and promotion status, and a sales/rolling-feature example
for one store-family series. The notebook does not include a standalone sales
histogram or a direct `describe()` summary.

The notebook's exploratory analysis reports 3,000,888 training rows and 28,512
test rows. It also reports 54 stores and 33 product families. These are dataset
observations, not model-performance results.

## 7. Data Cleaning and Preparation

The scripts provide CSV existence checks, type and missing-value diagnostics,
duplicate checks, date conversion, and daily-series alignment. The cleaning
helpers do not fill all missing cells indiscriminately.

For inserted daily rows, `reindex_daily_sales` leaves values missing by
default. The notebook explicitly passes zero for `sales` under the assumption
that the omitted Christmas dates represent days when stores were closed.
Other newly introduced fields remain missing and must be populated or handled
according to their own meaning and source.

## 8. Feature Engineering

Calendar features include year, month, day, day of week, and a weekend flag.
When `onpromotion` is available, `has_promotion` indicates whether its value is
greater than zero; missing promotion values remain missing. On complete daily
training histories, the scripts create `sales_lag_1`, `sales_lag_7`,
`sales_rolling_7`, and `sales_rolling_30`. Rolling averages exclude the current
day by shifting sales first.

## 9. Time Series Considerations

The notebook identifies four dates absent from every store-family series:
December 25 in 2013, 2014, 2015, and 2016. It reports 1,782 store-family
series and 7,128 added rows (four dates per series), increasing the training
row count from 3,000,888 to 3,008,016.

In the original notebook flow, lags and rolling features were created before
these dates were reindexed. Since `shift(7)` counts rows rather than elapsed
calendar days, it did not consistently mean “seven calendar days earlier.”
The project notebook and reusable workflow now complete each daily series and
set sales on inserted dates before recalculating the features. This zero-sales
policy is an assumption based on the shared Christmas gaps; validate it against
business context before using the data for forecasting.

The first 1, 7, or 30 observations per series naturally lack enough past
history for the corresponding features and remain `NaN`. No target-derived
features are created for the competition test horizon because future actual
sales are unavailable.

## 10. Key Findings

- Every one of the 1,782 store-family combinations has four missing dates, all
  on Christmas Day.
- Reindexing the daily series adds 7,128 rows; the notebook's code initially
  left the new rows' non-key fields missing.
- The notebook compares promotion status and temporal sales summaries and
  investigates how national, regional, and local holidays map to stores.
- Lag and rolling features must be recalculated after calendar completion to
  preserve their intended day-based meaning.

No predictive models were trained or evaluated in this repository.

## 11. Project Structure

```text
store-sales-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── store_sales_analysis.ipynb
├── src/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   └── feature_engineering.py
├── docs/
│   └── project_documentation.md
└── data/
    └── README.md
```

## 12. How to Run

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Download the CSV files described in [`data/README.md`](data/README.md), then
inspect them with:

```bash
python -m src.data_loading
```

Open `notebooks/store_sales_analysis.ipynb` in VS Code, select the `.venv`
Python kernel, and run the cells from top to bottom. The notebook expects the
CSV files in `data/`; it does not download data or request Kaggle credentials.
The two modules are also usable from Python:

```python
from src.data_loading import load_datasets
from src.data_cleaning import reindex_daily_sales
from src.feature_engineering import engineer_features

datasets = load_datasets()
daily_train = reindex_daily_sales(datasets["train"], missing_sales_value=0.0)
featured_train = engineer_features(daily_train)
```

## 13. Limitations

- The full data files are not included in this repository.
- The zero-sales interpretation for the four omitted Christmas dates is an
  explicit assumption, not a value supplied by the source CSVs.
- Holiday rules and other merged variables require domain review, especially
  for transferred events and overlapping holidays.
- This repository stops at EDA and data preparation; it has no forecasting
  model, validation strategy, or performance metrics.
- Outputs were cleared from the English-language notebook copy; run the cells
  locally with the CSV files to regenerate them.

## 14. Future Improvements

- Add automated tests against representative real-data samples.
- Revisit holiday-transfer handling and validate each regional/local mapping.
- Define a chronological validation protocol and leakage-safe test-horizon
  feature generation.
- Compare explicit treatment strategies for closed days and missing covariates.
- Add a forecasting model only after the preparation assumptions and evaluation
  design are agreed upon.
