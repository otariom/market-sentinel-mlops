import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from model_eval import evaluate_models


PROCESSED_DATA_PATH = "data/processed_data.csv"
REFERENCE_DATA_PATH = "data/reference_data.csv"
MODEL_DIR = "models"
PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "production_model.pkl")


def update_reference_data():
    """
    Overwrite reference_data.csv with the latest processed_data.csv
    after successful model promotion.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.to_csv(REFERENCE_DATA_PATH, index=False)
    print("Reference data updated successfully.")


def train_model():

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # -----------------------------
    # Feature / Target Split
    # -----------------------------
    target_column = "Revenue"   # <-- change this if your target is different

    # Drop identifier columns (IMPORTANT)
    drop_columns = [target_column]

    if "CustomerID" in df.columns:
        drop_columns.append("CustomerID")

    X = df.drop(columns=drop_columns)
    y = df[target_column]

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # MLflow Setup (Local File Store)
    # -----------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Market_Sentinel_Experiment")

    with mlflow.start_run():

        # -----------------------------
        # Train Model
        # -----------------------------
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        mlflow.xgboost.log_model(model, name="model")

    # -----------------------------
    # Ensure models folder exists
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save new candidate model temporarily
    new_model_path = os.path.join(MODEL_DIR, "candidate_model.pkl")
    joblib.dump(model, new_model_path)

    # -----------------------------
    # Model Comparison & Promotion
    # -----------------------------
    new_rmse = rmse
    print(f"New Model RMSE: {new_rmse}")

    if not os.path.exists(PRODUCTION_MODEL_PATH):
        print("No production model found. Promoting new model.")
        joblib.dump(model, PRODUCTION_MODEL_PATH)
        update_reference_data()
    else:
        production_model = joblib.load(PRODUCTION_MODEL_PATH)
        prod_predictions = production_model.predict(X_test)
        prod_rmse = mean_squared_error(y_test, prod_predictions) ** 0.5

        print(f"Production Model RMSE: {prod_rmse}")

        if new_rmse < prod_rmse:
            print("New model is better. Promoting to production.")
            joblib.dump(model, PRODUCTION_MODEL_PATH)
            update_reference_data()
        else:
            print("Production model is better. Keeping existing model.")

    print("Training cycle completed.")


if __name__ == "__main__":
    train_model()