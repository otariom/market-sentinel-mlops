# 📊 Market Sentinel: End-to-End MLOps Revenue Forecasting System

Market Sentinel is a production-grade MLOps ecosystem designed to forecast market revenue using high-frequency transactional data. This project demonstrates a **Self-Healing Machine Learning Lifecycle**, moving beyond static modeling into automated deployment, monitoring, and real-time inference.



---

## 🏗️ System Architecture
The system follows a modular, decoupled architecture to ensure scalability:

**Frontend (Streamlit)** → **Inference API (FastAPI)** → **Engine (XGBoost)** → **Monitoring (KS-Test)**

* **Data Layer:** Automated ingestion with frequency encoding for categorical markets.
* **Intelligence Layer:** Gradient Boosted Trees (XGBoost) optimized for tabular forecasting.
* **Validation Layer:** Pydantic-driven schema enforcement to ensure data integrity.
* **Stability Layer:** Statistical Drift Detection triggers automated retraining via CI/CD.

---

## 🌟 Key Features

### 🔹 MLOps & Automation
* **CI/CD Integration:** Automated testing and pipeline execution via **GitHub Actions**.
* **Containerization:** Fully **Dockerized** for "Write Once, Run Anywhere" deployment.
* **Model Promotion:** Logic-based "Champion-Challenger" evaluation before production deployment.

### 🔹 Monitoring & Reliability
* **Drift Detection:** Implements the **Kolmogorov-Smirnov Test** to detect shifts in feature distributions.
* **Self-Healing:** The system automatically identifies when the market "regime" changes and triggers a retraining cycle.

### 🔹 Business Intelligence
* **Interactive Dashboard:** A **Streamlit** interface allowing stakeholders to perform "What-If" scenario analysis.
* **Real-time Inference:** Low-latency predictions served through a RESTful FastAPI.

---

## 📈 Financial & Quant Engineering Context
For a BBA or Finance professional, this project serves as a foundation for:
* **Demand Elasticity:** Analyzing how `UnitPrice` fluctuations impact total `Revenue`.
* **Market Concentration:** Using Frequency Encoding to weight predictions based on regional market share.
* **Risk Mitigation:** Identifying "Out-of-Distribution" transactions that could signal market anomalies.



---

## 📂 Repository Structure
```text
├── .github/workflows/   # CI/CD Automation
├── src/
│   ├── app/             # FastAPI & Streamlit Logic
│   ├── data_ingestion.py# Feature Engineering & Cleaning
│   ├── drift_detection.py# Statistical Monitoring
│   ├── model_train.py   # XGBoost Training Pipeline
│   └── model_eval.py    # RMSE/MAE Evaluation
├── models/              # Serialized production models (.pkl)
├── Dockerfile           # Environment Containerization
└── requirements.txt     # Dependency Management
****


🚦 Quick Start
1. Local Setup
Bash
pip install -r requirements.txt
python src/data_ingestion.py
python src/model_train.py
2. Run API & UI
Bash
# Start Backend
uvicorn src.app.main:app --reload
# Start Frontend
streamlit run src/app/streamlit_app.py
3. Docker Deployment
Bash
docker build -t market-sentinel .
docker run -p 8000:8000 market-sentinel


🛠️ Technologies
Python | XGBoost | FastAPI | Streamlit | Docker | GitHub Actions | Scipy
