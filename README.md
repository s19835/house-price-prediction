# End-to-End Machine Learning Pipeline: House Price Prediction

## Project Overview

This project is aimed to build a complete end-to-end machine learning pipeline, covering the entire lifecycle of ML model, covering data collection and preprocessing to model training, hyperparameter tuning, and deployment. The pipeline will be implemented using Python by leveraging tools like pandas, Scikit-learn, XGBoost, and TensorFlow/PyTorch for modeling purposes, and Flask/FastAPI for deployment.

Using Kaggle’s House Prices: Advanced Regression Techniques dataset, we will develop a machine learning model to predict house prices based on various features such as square footage, number of rooms, and location.

The final trained model will be deployed as a web service using cloud platforms like Heroku or AWS, allowing real-time predictions through an API.

### Tools & Techniques

- Data Preprocessing: Pandas, NumPy, Scikit-learn
- Modeling: Scikit-learn, XGBoost, TensorFlow/PyTorch
- Visualization: Matplotlib, Seaborn
- API Development: Flask or FastAPI
- Deployment: Docker, Heroku, AWS/GCP
- Version Control: Git/GitHub

### Project Structure

```plaintext
End-to-End-ML-Pipeline/
.
├── LICENSE
├── README.md
├── analysis
│   ├── bivariate_analysis.py
│   ├── missing_value_analysis.py
│   ├── multivariate_analysis.py
│   └── univariate_analysis.py
├── config
│   └── config.yaml
├── data
│   ├── features
│   ├── processed
│   └── raw
│       ├── csvjson.json
│       ├── data_description.txt
│       ├── test.csv
│       └── train.csv
├── deployment
│   └── deployment_pipeline.py
├── notebooks
│   └── eda 006.ipynb
├── requirements.txt
├── run_deployment.py
├── run_pipline.py
├── sample_prediction.py
├── src
│   ├── data_inspection.py
│   ├── data_splitting.py
│   ├── feature_engineering.py
│   ├── handle_missing_values.py
│   ├── load_data.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── outlier_detection.py
├── steps
│   ├── data_splitting_step.py
│   ├── dynamic_importer.py
│   ├── feature_engineering_step.py
│   ├── load_data_step.py
│   ├── missing_value_handling_step.py
│   ├── model_building_step.py
│   ├── model_evaluation_step.py
│   ├── outlier_detection_step.py
│   ├── prediction_service_loader.py
│   └── predictor.py
├── tests
│   ├── outlier_detection_test.py
│   ├── test_data.py
│   └── test_model.py
└── training
    └── training_pipeline.py
```
