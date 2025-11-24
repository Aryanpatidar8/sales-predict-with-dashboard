"""Wrapper for preprocessing helpers now provided by `ml_utils`."""
from ml_utils import feature_engineer, build_preprocessor

__all__ = ["feature_engineer", "build_preprocessor"]

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "sample_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = feature_engineer(df)
    pre = build_preprocessor()
    X = df.drop(columns=["sales", "date"]).copy()
    pre.fit(X)
    print("Preprocessor ready")
