"""Load the Store Sales CSV files from the project's data directory."""

from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
}


def load_datasets(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all required CSV files, raising a clear error if any are missing."""
    root = Path(data_dir).expanduser()
    paths = {name: root / filename for name, filename in DATA_FILES.items()}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required dataset files were not found. Place the Kaggle CSV files "
            f"in {root}.\nMissing files:\n{missing_list}"
        )

    return {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }


def print_dataset_overview(datasets: Dict[str, pd.DataFrame]) -> None:
    """Print dimensions, data types, missing values, duplicates, and a sample."""
    for name, dataframe in datasets.items():
        print(f"\n=== {name} ===")
        print(f"Shape: {dataframe.shape}")
        print("\nData types:")
        print(dataframe.dtypes)
        print("\nMissing values:")
        print(dataframe.isna().sum())
        print(f"\nDuplicate rows: {dataframe.duplicated().sum()}")
        print("\nFirst five rows:")
        print(dataframe.head())


if __name__ == "__main__":
    print_dataset_overview(load_datasets())
