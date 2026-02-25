# 📡 MLOps Pipeline for Customer Churn prediction

An MLOps pipeline that automates the lifecycle of Telco churn prediction model. This project demonstrates a modular architecture handling **Data Ingestion**, **Validation**, **Training**, **Experiment Tracking (MLflow)**, **FastAPI** and **Dockerized Deployment** on Azure.

## 🚀 Executive Summary

This system implements a robust MLOps workflow where every stage is managed by a dedicated component:
* **Data Integrity:** Validates input data schema and quality using **Great Expectations**.
* **Experimentation:** Uses **MLflow** (with a local SQLite backend) to track hyperparameters, metrics, and artifacts.
* **Reproducibility:** Models are exported and "baked" into Docker images for stable, offline deployment.
* **Scalability:** Deployed as a scalable microservice using **FastAPI** and **Azure Container Instances (ACI)**.

---

## 📂 Project Structure

```bash
├── config.yaml                    # Master Config (Hyperparameters, Paths, SQL Connection)
├── Dockerfile                     # Blueprint for building the production Docker image
├── requirements.txt               # Python dependencies
├── mlflow.db                      # SQLite Database for MLflow Tracking (Local Registry)
├── mlruns                         # Folder storing MLflow artifacts (Models, Plots)
├── exported_model/                # (Generated) Stores the model artifacts for the Docker container
└── src/
    ├── data_ingestion.py          # Step 1: Loads raw data from source (CSV/DB)
    ├── data_validation.py           # Step 2: Validates schema using Great Expectations
    ├── data_preprocessing.py          # Step 3: Cleaning, Encoding, and Scaling
    ├── train.py                   # Step 4: Trains logistic_regression, RandomForest and XGBoost models and find the best model using Optuna
    ├── deploy.py                  # Deploy: Script to trigger deployment (ACR/ACI)
    ├── deploy_acr.py              # Deploy: Script to deploy on azure, used by deploy.py
    ├── serving.py                 # Core Logic: Smart Model Loading using Production mlflow model or exported model deopending on availability
    ├── app.py                     # API: FastAPI endpoints (/predict, /health, /train, /check_model)
    └── utils.py                   # Helper functions (Loading Config file)
```

## 🛠️ Prerequisites
* OS: Windows (Development) & Linux (Production)

* Python: 3.10

* Docker: Installed and running.

* Azure: Service Principal (SPN) credentials (for cloud deployment).



## 🚦 Execution Guide (Step-by-Step)
Follow these phases to execute the full pipeline.

### Phase 1: Data Pipeline & Training (Local)
* 1. Data Ingestion Loads the raw data(from the source defined in config.yaml to your raw data folder), validates it using pydantic and preprocesses it for model training.

```bash python src/data_ingestion.py ```
*2. Train the model using preprocessed data in the previous step. Store the best available model as /exported_model folder

```bash python src/train.py```
* 3. Verify in MLflow Open the UI to inspect runs, compare accuracy/AUC, and view artifacts.

```bash mlflow ui --backend-store-uri sqlite:///mlflow.db ```

## Model Lifecycle (Promotion)
Register: Code registers and moves the model to Staging in the training code.

Promote: This model is promoted to "Production" if it is better than the previously available model. We compare ROC_AUC metric

## Export & Docker Generation (CI/CD Build)
1. run deploy.py to deploy the model directly. This script follows 3 steps:
    a) Generate and test docker container in Linux server using below steps:
       i) Copy the files in the server using "pscp" command.
       ii) Create a docker image.
       iii) Run the docker container using the image generated.
       iv) Test if the container is working using test.py file.
    b) Deploy the model in Azure. The script follows the below steps in deploy_acr.py:
       i) Login and create Azure Registry and upload the tested image on azure.
       ii) Create Azure container instance(ACI) using ACR.
   c) This will return a public IP, which you can directly use to use the Churn prediction API


## 🔌 API Usage
Endpoint: POST /predict

Sample Request:

JSON
{
"customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

Sample Response:

JSON

{
  "input_data": {...}
  "prediction": [0],
}

### Use this using swagger UI means http://localhost:8100/docs

### steps to execute:
# Model Training and Deployment Workflow

## 1. Upload the Data
- Place the raw data in the `data/raw` folder.

## 2. Train the Model
- Run the `train.py` 

## 3. Run app.py using uvicorn
- uvicorn app:app --app-dir src --reload

## 5. Deploy Model for Docker
  - Run deploy.py file to deploy and get a public IP to access the API.
