"""Compatibility wrapper: provide `load_data` via `ml_utils`.

Existing scripts that import `data_loader.load_data` will continue to work.
"""
from ml_utils import load_data

__all__ = ["load_data"]

if __name__ == "__main__":
    from pathlib import Path
    df = load_data(Path(__file__).parent / "sample_sales.csv")
    print(df.head())
