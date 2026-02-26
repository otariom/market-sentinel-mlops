import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path="config/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_data_ingestion():
    config = load_config()

    raw_path = config["data"]["raw_data_path"]
    processed_path = config["data"]["processed_data_path"]

    df = pd.read_csv(raw_path)

    # Rename columns to match our pipeline expectations
    df = df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    })

    # Basic Cleaning
    df = df[df["Quantity"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates()

    # Feature Engineering
    # Feature Engineering
    df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=True,
    errors="coerce"
)

    df = df.dropna(subset=["InvoiceDate"])

    df["year"] = df["InvoiceDate"].dt.year
    df["month"] = df["InvoiceDate"].dt.month
    df["day"] = df["InvoiceDate"].dt.day
    df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
    df["hour"] = df["InvoiceDate"].dt.hour

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Encode Country using frequency encoding
    country_freq = df["Country"].value_counts().to_dict()
    df["Country_freq"] = df["Country"].map(country_freq)

    # Drop non-numeric columns
    df = df.drop(columns=[
        "InvoiceNo",
        "StockCode",
        "Description",
        "Country",
        "InvoiceDate"
    ])

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print("Processed data saved successfully.")


if __name__ == "__main__":
    run_data_ingestion()