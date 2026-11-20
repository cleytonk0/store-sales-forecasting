# Technical Documentation — Store Sales Forecasting

## 1. Project Context

This project organizes the exploratory analysis and data preparation for the
**Store Sales - Time Series Forecasting** challenge published by Corporación
Favorita on Kaggle. The target variable is `sales`, which records sales volume
by store, product family, and date. It does not represent product prices.

The notebook is the source of the documented analyses. Scripts in `src/`
separate data loading, quality checks, daily-series alignment, and feature
creation. This repository does not train or evaluate forecasting models.

## 2. What Is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis (EDA) is the process of understanding a dataset before
making decisions about data preparation or modeling. It helps identify the
dataset's structure, types, distributions, missing values, duplicates,
relationships between variables, and potential issues.

In Pandas, `DataFrame.shape` reports the number of rows and columns;
`DataFrame.dtypes` shows data types; `DataFrame.info()` provides a structural
summary; `isna().sum()` counts missing values; and `duplicated().sum()` counts
duplicate rows. `value_counts()` helps inspect categories such as holiday types
or promotion status. Although `describe()` is a common statistical summary, it
is not called directly in the original notebook.

The notebook also uses charts to examine oil prices, store geography, and event
types, and to compare sales by weekday, weekend, and promotion status. A
visualization is an exploratory tool; by itself, it does not demonstrate
causality.

## 3. What Are Time Series?

A time series is a set of observations ordered over time. In this project, each
combination of `store_nbr` and `family` forms a daily sales series. Ordering
matters: a previous observation can describe the past, but a future observation
must not be treated as available at prediction time.

A complete daily series has one row per day for each store-family combination
in the analyzed period. If dates are missing, position-based operations such as
`shift(7)` count records and do not necessarily represent seven calendar days.

## 4. How Pandas Organizes Data

Pandas primarily provides `Series` (a single column) and `DataFrame` (a table
with rows and columns). The notebook uses vectorized operations: instead of
iterating over rows manually, methods such as `groupby`, `merge`, `shift`, and
`rolling` operate on collections of records.

Examples:

- `pd.read_csv(path)` reads a CSV file into a DataFrame.
- `pd.to_datetime(df["date"])` converts date text to a datetime type.
- `df.loc[condition, columns]` filters rows and selects columns by label.
- `df.groupby(keys)` splits a table into groups with matching keys.
- `merge(..., how="left", validate="many_to_one")` enriches a table while
  preserving its rows and validates that the reference side does not contain
  duplicate matches.
- `sort_values(...)` sorts rows; `reset_index(drop=True)` creates a new
  numeric index without adding the old index as a column.

## 5. Why Data Cleaning Matters

Missing values, dates stored as text, duplicates, and inconsistent keys can
change totals, joins, and temporal features. The functions in
`src/data_cleaning.py` help:

1. Inspect data types, dimensions, missing values, and duplicates.
2. Convert dates without hiding parsing errors.
3. Identify missing calendar days.
4. Reindex each store-family series while preserving other columns and avoiding
   indiscriminate imputation.

Duplicates by key and date should be investigated before calculating features.
The reindexing routine fails explicitly if it finds more than one row for the
same series and date.

## 6. Missing Values and Christmas Dates

The notebook identifies 1,782 unique store-family combinations. All are missing
December 25, 2013, 2014, 2015, and 2016. That is four dates per series, for a
total of `1,782 × 4 = 7,128` rows to insert. Reindexing increases the row count
from 3,000,888 to 3,008,016.

An absence does not automatically mean zero. For these four specific dates,
their absence from every series is consistent with store closures on Christmas
Day. The reindexing function leaves `sales` missing by default; the notebook
explicitly passes `missing_sales_value=0.0` for those dates under the store
closure assumption. This decision should be validated against operational
context because it is not directly observed in the CSV.

Other columns in the new rows remain missing. For example, zero is not assigned
to `onpromotion`, oil prices, identifiers, or store attributes without
justification. Unobserved promotions remain missing; store attributes may be
filled by joining the store source; holiday indicators should be reconstructed
from the calendar. Each column requires a rule consistent with its meaning.

`ffill()` propagates the last known value, while `bfill()` propagates the next
known value. The notebook uses these methods for oil prices, but filling should
be limited to that variable and justified temporally. An imputation method
should not be applied to every column at once.

## 7. Integrating CSV Files

The workflow reads `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, and
`holidays_events.csv`. The integration demonstrated in the notebook uses:

- `store_nbr` to associate each store with its city, state, type, and cluster;
- `date` to associate daily oil prices;
- `date` and state/city to associate regional and local holidays;
- dates to consolidate national holidays.

Calendars may contain more than one event on the same date. Before a `merge`,
the notebook aggregates or removes duplicates in auxiliary tables and uses
`validate="many_to_one"` in relevant joins to prevent row multiplication.

Although the original text mentions transactions, the notebook code does not
read `transactions.csv` or integrate it into the final dataset. This file is
therefore not considered a variable in the prepared dataset.

## 8. Features and Feature Engineering

A *feature* is an explanatory variable organized to represent potentially
useful information. Feature engineering creates or transforms these variables
from available data.

### Calendar Features

The `.dt` accessor extracts calendar components from a datetime series:

- `.dt.year`: year;
- `.dt.month`: month;
- `.dt.day`: day of the month;
- `.dt.dayofweek`: day of the week, where Monday is 0 and Sunday is 6.

The `weekend` feature is 1 for Saturday/Sunday and 0 otherwise. `has_promotion`
is 1 when `onpromotion > 0`, 0 when a no-promotion observation exists, and
missing when `onpromotion` was not reported.

### Lag Features

A lag variable shifts earlier values within each series. `sales_lag_1`
represents the previous day's value; `sales_lag_7` retrieves the value from
seven days earlier. Grouping by `store_nbr` and `family` prevents one series'
history from contaminating another.

The functions sort by group and date before calling `shift`. They also require
consecutive daily intervals; if gaps exist, they require the series to be
reindexed first.

### Rolling Features

A rolling feature summarizes a historical window. `sales_rolling_7` and
`sales_rolling_30` calculate averages over the previous seven and thirty days.
Calling `shift(1)` before `rolling(...).mean()` excludes the current day's
sales. Setting `min_periods` to the window size leaves the result missing until
there is enough history.

The initial missing lag and moving-average values are expected: there is not
enough prior history at the beginning of each series.

## 9. Data Leakage and Chronological Order

Data leakage occurs when a feature uses information that would not be available
at prediction time. Including the current day's sales in a moving average used
to predict that same day is one example; therefore, the moving averages are
shifted by one row before calculation.

The original workflow had another issue: `sales_lag_1`, `sales_lag_7`,
`sales_rolling_7`, and `sales_rolling_30` were calculated before inserting the
four Christmas dates. With gaps, `shift(7)` meant seven previous records, not
necessarily seven calendar days. Reindexing afterward also created rows without
sales or temporal features. In this project copy, the gap diagnosis and
reindexing occur before temporal feature engineering; only the inserted rows'
`sales` values are set to zero, and features are calculated afterward.

Any future validation must also respect chronology: split training and
validation by date, and decide whether the task is one-step or multi-horizon
forecasting. For forecasts many days ahead, actual sales from intermediate
days will not be available as lag values.

## 10. Main Notebook Functions and Commands

| Command/function | Use in the notebook |
|---|---|
| `pd.read_csv()` | Loads CSV files into DataFrames. |
| `.shape`, `.dtypes`, `.info()` | Inspect dimensions and data types. |
| `.isna().sum()`, `.duplicated().sum()` | Count missing values and duplicates. |
| `pd.to_datetime()` | Converts dates before sorting and comparing periods. |
| `.value_counts()` | Summarizes categories, such as event types and promotion status. |
| `.loc[]` | Filters holidays by type, locale, or transferred status. |
| `.merge()` | Integrates store, oil, and calendar information with sales rows. |
| `.groupby()` | Isolates series and consolidates events by date/locale. |
| `.shift()` | Creates lags without mixing stores and families. |
| `.rolling().mean()` | Calculates historical window averages. |
| `pd.date_range()` / `.difference()` | Locates missing calendar days. |
| `MultiIndex.from_product()` / `.reindex()` | Creates the expected daily index and aligns rows. |
| `matplotlib.pyplot` / `seaborn` | Build the existing exploratory charts. |

The notebook's statistics mainly come from counts, chart averages, and
structural outputs; `describe()` is not applied directly.

## 11. How These Steps Prepare for a Future Model

A consistent, chronologically ordered dataset can be used as input to a future
forecasting model. Before that, it will still be necessary to define which
variables will be available at inference time, review holiday rules, decide how
to handle missing feature values, and evaluate strategies for the forecast
horizon. None of these choices replaces temporal validation, and this project
does not produce model metrics.

## 12. Issues Identified in the Original Notebook

- Initial cells requested KaggleHub authentication/downloads, but later read
  files relative to the current directory instead of using the downloaded
  path. The project copy uses local `data/` files and does not request
  credentials.
- A test-dataset diagnostic cell queried `oil` before it was read and assigned
  oil dates to the test table. It was corrected to inspect dates from
  `test.csv` itself.
- The original lag calculation preceded daily reindexing. The project copy
  reorders these steps and recalculates features after inserting missing days.
- The original reindexing left every field missing in new rows; now only the
  inserted sales values are set to zero under the documented assumption.
- Some integration and visualization cells were repeated; rerunning successive
  merges could create columns with `_x` and `_y` suffixes. The modules provide
  separate functions for a reproducible workflow.

## 13. References

- [Store Sales - Time Series Forecasting competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [Pandas documentation](https://pandas.pydata.org/docs/)
