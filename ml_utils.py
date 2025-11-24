"""Combined ML utilities: data loading, EDA, preprocessing, training, and prediction.

This module merges small helpers from `data_loader.py`, `eda.py`, `preprocessing.py`,
`predict.py`, and `train.py` so the project has a single place for ML-related helpers.
It is intentionally conservative: functions keep the same signatures used elsewhere.
"""
from pathlib import Path
import pandas as pd
import joblib

# EDA imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

# Preprocessing and modeling imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["date", "store", "item", "price"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_eda(df: pd.DataFrame, outdir: str | Path = "outputs"):
    """Run a few basic EDA plots and save to `outdir` if plotting libs are available."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if plt is None or sns is None:
        print("Plotting libraries not available; skipping EDA plots")
        return

    plt.figure(figsize=(8, 4))
    sns.histplot(df["sales"], kde=True)
    plt.title("Sales Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "sales_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(x="date", y="sales", data=df.sort_values("date"))
    plt.title("Sales over Time")
    plt.tight_layout()
    plt.savefig(outdir / "sales_over_time.png")
    plt.close()


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    return df


def build_preprocessor(numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = ["price", "promotion", "day", "month", "weekday"]
    if categorical_features is None:
        categorical_features = ["store", "item"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def train(data_path: str, artifacts_dir: str = "artifacts"):
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    try:
        run_eda(df, Path(artifacts_dir) / "outputs")
    except Exception:
        pass

    df = feature_engineer(df)
    X = df.drop(columns=["sales", "date"]).copy()
    y = df["sales"].values

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.3f}, R2: {r2:.3f}")

    joblib.dump(model, artifacts / "model.joblib")
    joblib.dump(preprocessor, artifacts / "preprocessor.joblib")
    print(f"Saved model and preprocessor to {artifacts.resolve()}")


def load_artifacts(artifacts_dir: str = "artifacts"):
    artifacts = Path(artifacts_dir)
    model = joblib.load(artifacts / "model.joblib")
    preprocessor = joblib.load(artifacts / "preprocessor.joblib")
    return model, preprocessor


def predict_single(input_dict: dict, model, preprocessor):
    df = pd.DataFrame([input_dict])
    df["date"] = pd.to_datetime(df["date"])
    df = feature_engineer(df)
    X = df.drop(columns=["date"]).copy()
    X_proc = preprocessor.transform(X)
    pred = model.predict(X_proc)
    return float(pred[0])


def test_local_prediction(payload: dict, artifacts_dir: str = "artifacts"):
    model, pre = load_artifacts(artifacts_dir)
    return predict_single(payload, model, pre)
