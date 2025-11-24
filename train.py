import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from ml_utils import train as train_main


def train(data_path: str, artifacts_dir: str = "artifacts"):
    # Delegate to the combined `ml_utils.train` implementation
    return train_main(data_path, artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts output dir")
    args = parser.parse_args()
    train_main(args.data, args.artifacts)
