import os
import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def load_config(config_path="config/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_models(new_model_path, reference_data_path, target_col):
    df = pd.read_csv(reference_data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    new_model = joblib.load(new_model_path)

    new_preds = new_model.predict(X)
    new_rmse = np.sqrt(mean_squared_error(y, new_preds))

    print(f"New Model RMSE: {new_rmse}")

    # If no old model exists → automatically promote
    if not os.path.exists("models/production_model.pkl"):
        print("No production model found. Promoting new model.")
        return True

    old_model = joblib.load("models/production_model.pkl")
    old_preds = old_model.predict(X)
    old_rmse = np.sqrt(mean_squared_error(y, old_preds))

    print(f"Old Model RMSE: {old_rmse}")

    return new_rmse < old_rmse