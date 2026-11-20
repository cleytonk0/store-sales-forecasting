"""Data-quality checks and daily sales-series alignment helpers."""

from typing import Dict, Sequence

import pandas as pd


def data_quality_report(dataframe: pd.DataFrame) -> Dict[str, object]:
    """Return basic structural and missing-data diagnostics for a DataFrame."""
    report: Dict[str, object] = {
        "shape": dataframe.shape,
        "dtypes": dataframe.dtypes,
        "missing_values": dataframe.isna().sum(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }
    if "date" in dataframe.columns:
        parsed_dates = pd.to_datetime(dataframe["date"], errors="coerce")
        report["invalid_dates"] = int(
            (dataframe["date"].notna() & parsed_dates.isna()).sum()
        )
        valid_dates = parsed_dates.dropna()
        report["date_range"] = (
            (valid_dates.min(), valid_dates.max()) if not valid_dates.empty else None
        )
        if not valid_dates.empty:
            expected_dates = pd.date_range(
                valid_dates.min(), valid_dates.max(), freq="D"
            )
            report["missing_calendar_dates"] = len(
                expected_dates.difference(pd.DatetimeIndex(valid_dates.unique()))
            )

    series_keys = ["store_nbr", "family", "date"]
    if set(series_keys).issubset(dataframe.columns):
        report["duplicate_series_date_rows"] = int(
            dataframe.duplicated(series_keys).sum()
        )

    if "sales" in dataframe.columns:
        numeric_sales = pd.to_numeric(dataframe["sales"], errors="coerce")
        report["non_numeric_sales"] = int(
            (dataframe["sales"].notna() & numeric_sales.isna()).sum()
        )
        report["negative_sales"] = int(numeric_sales.lt(0).sum())

    return report


def convert_date_column(
    dataframe: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return a copy with the requested date column parsed as datetimes."""
    if date_column not in dataframe.columns:
        raise KeyError(f"Date column {date_column!r} is not present.")

    result = dataframe.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    return result


def find_missing_dates(
    dataframe: pd.DataFrame,
    date_column: str = "date",
) -> pd.DatetimeIndex:
    """Find dates absent from the dataset's global daily calendar."""
    if date_column not in dataframe.columns:
        raise KeyError(f"Date column {date_column!r} is not present.")
    if dataframe.empty:
        return pd.DatetimeIndex([])

    dates = pd.to_datetime(dataframe[date_column], errors="raise").dropna()
    if dates.empty:
        return pd.DatetimeIndex([])

    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    return expected.difference(pd.DatetimeIndex(dates.unique()))


def reindex_daily_sales(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str] = ("store_nbr", "family"),
    date_column: str = "date",
    sales_column: str = "sales",
    missing_sales_value: float | None = None,
) -> pd.DataFrame:
    """Add missing daily rows within each series, filling only inserted sales.

    By default, inserted sales remain missing. Pass ``missing_sales_value=0``
    only when domain knowledge supports interpreting absent dates as zero-sales
    observations. Other columns remain missing and existing missing sales
    values are not changed.
    """
    keys = list(key_columns)
    if not keys:
        raise ValueError("At least one series key column is required.")

    required = [*keys, date_column, sales_column]
    missing_columns = [column for column in required if column not in dataframe]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    if dataframe.empty:
        return dataframe.copy()

    result = convert_date_column(dataframe, date_column)
    if result[[*keys, date_column]].isna().any().any():
        raise ValueError("Series keys and dates cannot contain missing values.")
    if result.duplicated([*keys, date_column]).any():
        raise ValueError(
            "Duplicate rows were found for a series and date; resolve them "
            "before reindexing."
        )

    expanded_groups = []
    for group_values, group in result.groupby(keys, sort=False, dropna=False):
        if len(keys) == 1:
            group_values = (group_values,)

        group = group.sort_values(date_column, kind="stable").set_index(date_column)
        full_dates = pd.date_range(group.index.min(), group.index.max(), freq="D")
        inserted_dates = ~full_dates.isin(group.index)
        expanded = group.reindex(full_dates)

        for key, value in zip(keys, group_values):
            expanded[key] = value
        expanded.index.name = date_column
        if missing_sales_value is not None:
            expanded.loc[inserted_dates, sales_column] = missing_sales_value
        expanded_groups.append(expanded.reset_index())

    return (
        pd.concat(expanded_groups, ignore_index=True)
        .sort_values([*keys, date_column], kind="stable")
        .reset_index(drop=True)
    )
