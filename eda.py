"""Wrapper for `ml_utils.run_eda` to preserve API compatibility."""
from ml_utils import run_eda

__all__ = ["run_eda"]

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "sample_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    run_eda(df, Path(__file__).parent.parent / "outputs")
