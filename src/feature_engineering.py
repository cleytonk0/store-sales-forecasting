"""Calendar, promotion, lag, and rolling features for daily sales data."""

from typing import Sequence

import pandas as pd


def add_temporal_features(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    promotion_column: str = "onpromotion",
) -> pd.DataFrame:
    """Add calendar fields and a nullable promotion indicator when available."""
    if date_column not in dataframe.columns:
        raise KeyError(f"Date column {date_column!r} is not present.")

    result = dataframe.copy()
    dates = pd.to_datetime(result[date_column], errors="raise")
    if dates.isna().any():
        raise ValueError(f"Date column {date_column!r} contains missing values.")

    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["day_of_week"] = dates.dt.dayofweek
    result["weekend"] = (dates.dt.dayofweek >= 5).astype("int8")

    if promotion_column in result.columns:
        has_promotion = pd.Series(pd.NA, index=result.index, dtype="Int8")
        observed = result[promotion_column].notna()
        has_promotion.loc[observed] = (
            result.loc[observed, promotion_column].gt(0).astype("int8")
        )
        result["has_promotion"] = has_promotion

    return result


def add_sales_history_features(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str] = ("store_nbr", "family"),
    date_column: str = "date",
    sales_column: str = "sales",
) -> pd.DataFrame:
    """Create leakage-aware lags and rolling means on complete daily series."""
    keys = list(key_columns)
    required = [*keys, date_column, sales_column]
    missing_columns = [column for column in required if column not in dataframe]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    result = dataframe.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    if result[[*keys, date_column]].isna().any().any():
        raise ValueError("Series keys and dates cannot contain missing values.")
    if result.duplicated([*keys, date_column]).any():
        raise ValueError("Each series must contain at most one row per date.")

    result = result.sort_values([*keys, date_column], kind="stable").reset_index(
        drop=True
    )
    intervals = result.groupby(keys, sort=False)[date_column].diff().dropna()
    if not intervals.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Daily date gaps remain. Reindex each series with "
            "reindex_daily_sales() before creating lag or rolling features."
        )

    sales_by_series = result.groupby(keys, sort=False)[sales_column]
    result["sales_lag_1"] = sales_by_series.shift(1)
    result["sales_lag_7"] = sales_by_series.shift(7)
    result["sales_rolling_7"] = sales_by_series.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean()
    )
    result["sales_rolling_30"] = sales_by_series.transform(
        lambda values: values.shift(1).rolling(window=30, min_periods=30).mean()
    )
    return result


def engineer_features(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str] = ("store_nbr", "family"),
    date_column: str = "date",
    sales_column: str = "sales",
    promotion_column: str = "onpromotion",
) -> pd.DataFrame:
    """Add calendar/promotion fields and, when sales exist, history features."""
    result = add_temporal_features(
        dataframe,
        date_column=date_column,
        promotion_column=promotion_column,
    )
    if sales_column in result.columns:
        result = add_sales_history_features(
            result,
            key_columns=key_columns,
            date_column=date_column,
            sales_column=sales_column,
        )
    return result
