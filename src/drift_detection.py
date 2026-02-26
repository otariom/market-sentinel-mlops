import yaml
import pandas as pd
import sys
from scipy.stats import ks_2samp


def load_config(config_path: str = "config/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_drift_detection():
    config = load_config()

    reference_path = config["data"]["reference_data_path"]
    processed_path = config["data"]["processed_data_path"]
    min_drifted_features = config["drift"]["min_drifted_features"]

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(processed_path)

    numeric_columns = reference_df.select_dtypes(include="number").columns

    drifted_features = 0

    for col in numeric_columns:
        stat, p_value = ks_2samp(reference_df[col], current_df[col])

        # If p-value < 0.05 → distributions are statistically different
        if p_value < 0.05:
            drifted_features += 1

    dataset_drift = drifted_features > 0

    print(f"Dataset Drift Detected: {dataset_drift}")
    print(f"Number of Drifted Features: {drifted_features}")

    if dataset_drift and drifted_features >= min_drifted_features:
        print("Drift threshold exceeded. Retraining required.")
        sys.exit(1)
    else:
        print("No significant drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    run_drift_detection()