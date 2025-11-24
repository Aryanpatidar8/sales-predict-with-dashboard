"""Create minimal dummy preprocessor and model and save to artifacts/.

This creates files expected by `predict.py`: `artifacts/model.joblib` and
`artifacts/preprocessor.joblib`. The files are small and only intended for
local testing of the UI while you don't have real trained artifacts.
"""
from pathlib import Path
import random
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


def make_sample_df(n=300):
    stores = ["StoreA", "StoreB", "StoreC"]
    items = ["Item1", "Item2", "Item3"]
    rows = []
    start = datetime(2023, 1, 1)
    for i in range(n):
        d = start + timedelta(days=random.randrange(0, 365))
        store = random.choice(stores)
        item = random.choice(items)
        promo = random.choice([0, 1])
        price = round(random.uniform(5, 50), 2)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "store": store,
            "item": item,
            "promotion": promo,
            "price": price,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    X = df.drop(columns=["date"]).copy()
    return X


def build_and_save(artifacts_dir: str = "artifacts"):
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(exist_ok=True)

    X = make_sample_df(400)

    cat_cols = ["store", "item"]
    num_cols = ["promotion", "price", "day", "month", "weekday"]

    # Use `sparse_output=False` for newer scikit-learn, fall back to `sparse` name if needed
    try:
        preproc_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        preproc_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", preproc_encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Fit preprocessor
    preprocessor.fit(X)

    # Build a simple target: price * factor + promotion * bias + weekday effect
    price = X["price"].to_numpy()
    promo = X["promotion"].to_numpy()
    weekday = X["weekday"].to_numpy()
    y = price * 1.5 + promo * 20 + (weekday % 7) * 0.5 + np.random.normal(0, 5, size=price.shape)

    # Transform X and fit a simple linear model
    X_trans = preprocessor.transform(X)
    model = LinearRegression()
    model.fit(X_trans, y)

    # Save artifacts
    joblib.dump(model, artifacts / "model.joblib")
    joblib.dump(preprocessor, artifacts / "preprocessor.joblib")
    print("Wrote:")
    print(" -", artifacts / "model.joblib")
    print(" -", artifacts / "preprocessor.joblib")


if __name__ == "__main__":
    build_and_save()
