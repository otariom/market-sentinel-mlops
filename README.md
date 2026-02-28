📊 Market Sentinel: End-to-End MLOps Pipeline
Market Sentinel is a production-ready machine learning system designed for market revenue prediction. This project demonstrates a complete MLOps lifecycle, moving beyond a static notebook to a fully automated, monitorable, and deployable service.

🛠️ System Architecture
The project is built as a modular pipeline to ensure scalability and maintainability:
Data Ingestion & Engineering: Automated cleaning and feature engineering of raw market datasets.
Model Training: High-performance regression using XGBoost with experiment tracking.
Statistical Drift Detection: Monitoring data health using the Kolmogorov-Smirnov (KS) Test to identify shifts in input distributions.
Model Evaluation & Promotion: A "Champion-Challenger" logic that only deploys the new model if it statistically outperforms the current production version.
API Serving: Real-time inference delivered via FastAPI with strict type validation.

🚀 Key Features
Automated Retraining: The pipeline triggers a fresh training cycle automatically if data drift is detected.
Containerized Deployment: Packaged with Docker for consistent execution across dev, staging, and production.
CI/CD Integration: Automated workflows via GitHub Actions for testing and pipeline orchestration.

Type Safety: Uses Pydantic models to ensure data integrity during API requests.

💻 Tech Stack
Modeling: XGBoost, Scikit-Learn
Statistical Testing: SciPy (KS-Test)
API Framework: FastAPI, Uvicorn
Data Handling: Pandas, NumPy
DevOps: Docker, GitHub Actions, Joblib

🚦 Getting Started
1. Prerequisites
Python 3.10+
Docker (Optional, for containerization)
2. Installation
Bash
git clone https://github.com/otariom/market-sentinel-mlops.git
cd market-sentinel-mlops
pip install -r requirements.txt
3. Running the Pipeline
To run the full cycle (Ingestion -> Drift Check -> Train -> Deploy):

Bash
python src/data_ingestion.py
python src/drift_detection.py
# If drift is detected, the pipeline triggers model_train.py
4. Serving the Model
Bash
uvicorn main:app --reload
Once running,
access the interactive API documentation at http://127.0.0.1:8000/docs.
