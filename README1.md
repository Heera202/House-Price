# 🏠 End-to-End House Price Prediction with ZenML & MLflow

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ZenML](https://img.shields.io/badge/ZenML-MLOps-432E54?style=for-the-badge&logo=zenml&logoColor=white)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A production-ready machine learning pipeline for predicting house prices, built with modern MLOps practices using ZenML for workflow orchestration and MLflow for experiment tracking and model deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Model Workflow](#-model-workflow)
- [Machine Learning Features](#-machine-learning-features)
- [Deployment & Serving](#-deployment--serving)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)

---

## 🎯 Overview

This project demonstrates an end-to-end machine learning pipeline for house price prediction, incorporating best practices in MLOps, data engineering, and model deployment. The system predicts house prices based on various features including size, location, condition, quality, and year built.

**Key Objectives:**
- 🎯 Accurate price prediction using regression models
- 🔄 Automated, reproducible ML workflows
- 📊 Comprehensive experiment tracking
- 🚀 Production-ready REST API deployment

---

## 📊 Dataset

**Source:** Kaggle

**Description:** The dataset contains comprehensive housing features including:

- 💰 **SalePrice** - Target variable (house sale price)
- 📅 **YearBuilt** - Year of construction
- 📐 **LotArea** - Lot size in square feet
- ⭐ **OverallQual** - Overall material and finish quality
- 🛏️ **BedroomAbvGr** - Number of bedrooms above ground
- 🏘️ **Neighborhood** - Physical location within city limits
- And many more features...

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📥 **Data Ingestion** | Automated data loading with factory pattern |
| 🧹 **Data Cleaning** | Robust preprocessing & missing value handling |
| 📈 **EDA** | Comprehensive exploratory data analysis |
| ⚙️ **Feature Engineering** | Multiple transformation techniques |
| 🤖 **Model Training** | Hyperparameter tuning with ElasticNet |
| 📊 **Model Evaluation** | Multi-metric assessment (MAE, MSE, R²) |
| 🚀 **MLflow Deployment** | REST API for real-time predictions |
| 🔄 **Dual Inference** | Real-time & batch prediction modes |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logo=matplotlib&logoColor=white) |
| **ML Framework** | ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white) |
| **MLOps** | ![MLflow](https://img.shields.io/badge/-MLflow-0194E2?logo=mlflow&logoColor=white) ![ZenML](https://img.shields.io/badge/-ZenML-432E54?logo=zenml&logoColor=white) |
| **API** | ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white) |

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ZenML Pipeline Orchestration              │
├─────────────────────────────────────────────────────────────┤
│  Data Ingestion → Cleaning → EDA → Feature Engineering →    │
│  Training → Evaluation → Deployment                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Experiment Tracking                │
│  • Model Versioning  • Metrics Logging  • Artifact Storage   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      REST API Serving                        │
│         Real-time Predictions & Batch Inference              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Model Workflow

### 1️⃣ Data Ingestion
- 🏭 **Factory Pattern Implementation** - Automated ingester selection by file type
- 🧩 **Extensible Architecture** - Easy to add new file format support
- 📦 **ZIP File Processing** - Automatic extraction and CSV detection
- 🎨 **OOP Design** - Clean, maintainable code with abstraction and inheritance

### 2️⃣ Data Cleaning
- ✂️ **Flexible Splitting Strategies** - Train-test split & K-Fold cross-validation
- 🎯 **Strategy Pattern Architecture** - Easily switch between splitting methods
- 🧼 **Data Quality Checks** - Automated validation and consistency checks

### 3️⃣ Exploratory Data Analysis (EDA)
- 📊 **Distribution Analysis** - Statistical summaries and visualizations
- 🔍 **Correlation Studies** - Feature relationship identification
- 📈 **Trend Detection** - Temporal and categorical patterns

### 4️⃣ Feature Engineering
- 📐 **Multiple Transformation Techniques**
  - Log transformation
  - Standard Scaling
  - Min-Max Scaling
  - One-Hot Encoding
- 📉 **Skewness Handling** - Logarithmic transformation for normalized distributions
- 🔌 **Modular Design** - Plugin-based architecture for new transformations

### 5️⃣ Missing Value Handling
- 🧠 **Smart Imputation** - Mean, median, mode, and constant value filling
- ⚙️ **Configurable Thresholds** - Flexible missing value removal criteria
- 🛡️ **Production-Ready** - Robust error handling and logging

### 6️⃣ Outlier Detection & Treatment
- 📊 **Statistical Methods** - Z-Score and IQR-based outlier detection
- 🎯 **Multiple Handling Strategies** - Removal or capping of anomalies
- 📉 **Visualization Tools** - Automated boxplot generation for analysis

---

## 🤖 Machine Learning Features

### 🎓 Model Training & Architecture
- 📈 **ElasticNet Regression** - Regularized linear model with L1/L2 penalty
- ⚡ **Automated Preprocessing** - Intelligent handling of numerical & categorical features
- 📝 **MLflow Integration** - Complete experiment tracking and model versioning
- 🔄 **Pipeline Deployment** - End-to-end reproducible training workflows

### 📊 Model Evaluation & Validation
- 📏 **Comprehensive Metrics** - MAE, MSE, R², and custom evaluation strategies
- ✅ **Data Integrity Checks** - NaN detection and validation in test data
- 🎯 **Strategy Pattern** - Extensible evaluation framework for different model types
- 📈 **Production Monitoring** - Ready for model performance tracking

### 🎯 Model Serving & Management
- 🏷️ **Version Control** - Production model versioning with ZenML registry
- 📦 **Artifact Loading** - Seamless retrieval of trained pipelines
- 🚀 **CI/CD Ready** - Integration with deployment pipelines
- 🔄 **Reproducible Inference** - Consistent preprocessing during serving

---

## 🚀 Deployment & Serving

### 1️⃣ Continuous Deployment Pipeline
- 🤖 **Automated Model Deployment** - Train → Validate → Deploy in single workflow
- 🌐 **MLflow Model Serving** - Production-ready REST API with configurable workers
- ⚡ **Zero-Downtime Updates** - Seamless model version switching
- 🚦 **Deployment Gates** - Automated quality checks before promotion

### 2️⃣ Real-time Inference API
- 🔌 **RESTful Endpoints** - Standard HTTP API for predictions
- 📄 **JSON I/O Format** - Simple request/response structure
- 📊 **Scalable Serving** - Multi-worker architecture for high throughput
- 💓 **Health Monitoring** - Built-in health checks and metrics

### 3️⃣ Batch Inference Pipeline
- 📦 **Large-scale Predictions** - Efficient processing of multiple records
- 🔍 **Service Discovery** - Automatic detection of deployed models
- 📥 **Flexible Data Sources** - Support for various data import methods
- ⏰ **Scheduled Execution** - Ready for cron jobs and automated workflows

### 4️⃣ Model Serving Infrastructure
- 💻 **Local Development** - MLflow local server for testing
- ☁️ **Production Ready** - Easy migration to cloud platforms
- 🔧 **Service Management** - Start/stop/status monitoring capabilities
- ⚖️ **Load Balancing** - Configurable worker processes

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1️⃣ **Clone the Repository**
```bash
git clone <repository-url>
cd house-price-prediction
```

2️⃣ **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Initialize ZenML Stack

1️⃣ **Register MLflow Tracker**
```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
```

2️⃣ **Register Stack**
```bash
zenml stack register my_mlflow_stack -o default -a default -e mlflow_tracker
```

3️⃣ **Verify Stack Configuration**
```bash
zenml stack describe
```

### Run the Pipeline

**Train, Deploy, and Run Inference**
```bash
python run_deployment.py
```

### Test the API

**Make Real-time Predictions**
```bash
python sample_predict.py
```

**View Experiments in MLflow UI**
```bash
mlflow ui --backend-store-uri $(zenml mlflow get-tracking-uri)
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📈 Model Performance

| Model | MSE | R² Score | Status |
|-------|-----|----------|--------|
| **Linear Regression** | 403,589,797.47 | 0.9121 | ✅ Baseline |
| **ElasticNet** | 445,187,199.24 | 0.9030 | ✅ Production |

**Key Insights:**
- 🎯 Both models achieve excellent R² scores above 0.90
- 📊 Linear Regression shows slightly better performance
- ⚡ ElasticNet provides better generalization with regularization
- 🔄 Trade-off between accuracy and model complexity

---

## 📚 API Documentation

### Prediction Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "data": {
    "LotArea": 8450,
    "YearBuilt": 2003,
    "OverallQual": 7,
    "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr"
  }
}
```

**Response:**
```json
{
  "prediction": 208500.00,
  "model_version": "v1.0.0",
  "timestamp": "2025-10-27T10:30:00Z"
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Name** - *Initial work*

---

## 🙏 Acknowledgments

- Kaggle for providing the house price dataset
- ZenML team for the excellent MLOps framework
- MLflow community for model tracking and deployment tools
- scikit-learn for machine learning algorithms

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ using ZenML and MLflow

</div>
