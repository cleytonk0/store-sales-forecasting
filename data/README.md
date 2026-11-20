# Dataset

This project uses the **Store Sales - Time Series Forecasting** competition
dataset published by Corporación Favorita on Kaggle:

<https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data>

## Required files

Download and place these files directly in this `data/` directory:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`

`transactions.csv` is not loaded or integrated by the notebook code, so it is
not required for the project workflow.

## Getting access

1. Sign in to Kaggle and open the competition data page.
2. Accept the competition rules if Kaggle asks you to do so.
3. Download the required CSV files and copy them into `data/`.
4. Do not commit the CSV files, Kaggle tokens, or `kaggle.json`.

The project reads files locally and does not require Kaggle API credentials.
See the repository's `README.md` for installation and execution instructions.
