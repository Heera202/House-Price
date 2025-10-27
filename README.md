### END TO END HOUSE PRICE PREDICTION WITH ZENML  AND MLFLOW ###

The project is based on building end to end machine learning pipeline for a house price prediction system.

The pipeline is build using ZenML, a MLOps framework that helps in managing machine learning workflows and MLflow, which is used for experiment tracking and model deployment.

The primary goal of this project is to predict house price based on various features such as size, location, condition, quality and year_build

### DATASET ###
1. Data Collection 
  Kaggle 
2. Descripton: Contains housing features like such as 
  SalePrice, YearBuild, LoTArea, Overall Qual, Bedroom AbvGr,    Neighbour etc

### FEATURES ###
✅ Data Ingestion 

✅ Data Cleaning & Preprocessing

✅ Exploratory Data Analysis (EDA)

✅ Feature Engineering

✅ Model Training & Hyperparameter Tuning

✅ Model Evaluation

✅ Local MLFlow Deployment using REST-API

✅ Dual Inference Modes


### TECH STACK ###
Python, Pandas, Numpy, Matplotlib, Scikit-Learn, Mlflow, Mlops, FastAPI

### MODEL WORKDFLOW ###
1. Data Ingestion
   - **Factory Pattern Implementation**: Automated ingester selection by file type
- **Extensible Architecture**: Easy to add new file format support
- **ZIP File Processing**: Automatic extraction and CSV detection
- **OOP Design**: Clean, maintainable code with abstraction and inheritance

2. Data Cleaning
**Flexible Splitting Strategies**: Train-test split & K-Fold cross-validation
- **Strategy Pattern Architecture**: Easily switch between splitting methods

3. EDA 

4. Feature Engineering
- **Multiple Transformation Techniques**: Log, Standard Scaling, Min-Max Scaling, One-Hot Encoding
- **Skewness Handling**: Logarithmic transformation for normalized distributions
- **Modular Design**: Plugin-based architecture for new transformations
Missing Value Handling
- **Smart Imputation**: Mean, median, mode, and constant value filling
- **Configurable Thresholds**: Flexible missing value removal criteria
- **Production-Ready**: Robust error handling and logging

Outlier Detection & Treatment
- **Statistical Methods**: Z-Score and IQR-based outlier detection
- **Multiple Handling Strategies**: Removal or capping of anomalies
- **Visualization Tools**: Automated boxplot generation for analysis


#### Machine Learning Features ####

Model Training & Architecture
- **ElasticNet Regression**: Regularized linear model with L1/L2 penalty
- **Automated Preprocessing**: Intelligent handling of numerical & categorical features
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Pipeline Deployment**: End-to-end reproducible training workflows

Model Evaluation & Validation
- **Comprehensive Metrics**: MAE, MSE, R², and custom evaluation strategies
- **Data Integrity Checks**: NaN detection and validation in test data
- **Strategy Pattern**: Extensible evaluation framework for different model types
- **Production Monitoring**: Ready for model performance tracking

Model Serving & Management
- **Version Control**: Production model versioning with ZenML registry
- **Artifact Loading**: Seamless retrieval of trained pipelines
- **CI/CD Ready**: Integration with deployment pipelines
- **Reproducible Inference**: Consistent preprocessing during serving

#### Deployment & Serving Features ####

Continuous Deployment Pipeline
- **Automated Model Deployment**: Train → Validate → Deploy in single workflow
- **MLflow Model Serving**: Production-ready REST API with configurable workers
- **Zero-Downtime Updates**: Seamless model version switching
- **Deployment Gates**: Automated quality checks before promotion

Real-time Inference API
- **RESTful Endpoints**: Standard HTTP API for predictions
- **JSON I/O Format**: Simple request/response structure
- **Scalable Serving**: Multi-worker architecture for high throughput
- **Health Monitoring**: Built-in health checks and metrics

Batch Inference Pipeline
- **Large-scale Predictions**: Efficient processing of multiple records
- **Service Discovery**: Automatic detection of deployed models
- **Flexible Data Sources**: Support for various data import methods
- **Scheduled Execution**: Ready for cron jobs and automated workflows

Model Serving Infrastructure
- **Local Development**: MLflow local server for testing
- **Production Ready**: Easy migration to cloud platforms
- **Service Management**: Start/stop/status monitoring capabilities
- **Load Balancing**: Configurable worker processes

### RUN LOCALLY ###
1. Create Virtual Environment
  python -m venv venv
  source venv/bin/activate   # for Linux/Mac
  venv\Scripts\activate      # for Windows

2. Install Dependencies
  pip install -r requirements.txt

3. Run
   
   
   ### Initialize mlflow tracker
       zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   
   ### Register mlflow in stack
        zenml stack register my_mlflow_stack -o default -a default -a default -e mlflow_tracker

   ### Check the stack table
       zenml stack describe      

   ### Train, deploy, and run inference
        python run_deployment.py
   
5. Test the API
  ### Make real-time predictions
    python smaple_predict.py

  ### View experiments in MLflow UI
    mlflow ui --backend-store-uri $(zenml mlflow get-tracking-uri)
    Open http://localhost:5000



