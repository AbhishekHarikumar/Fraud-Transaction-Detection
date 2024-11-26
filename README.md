# Fraud-Transaction-Detection

<h2 align="left">Description</h2>

Fraud detection is a critical need in today's financial and transactional ecosystems, where digital transactions are increasingly susceptible to sophisticated fraudulent activities. This project presents a comprehensive machine learning pipeline designed to detect fraudulent transactions
using a dataset stored in MongoDB. The pipeline encompasses data extraction, transformation, and advanced preprocessing techniques, ensuring high-quality input for the model. Key steps include domain-specific feature engineering, such as time-based attributes and indicators for
transactional anomalies, outlier detection using capping methods, and addressing class imbalance with SMOTE. An XGBoost classifier, renowned for its performance on imbalanced datasets and high-dimensional data, is employed as the core predictive model. The pipeline follows a
structured workflow, including stratified train-test splitting, normalization using standard scaling, and robust model evaluation. Post-training, the best-performing model is serialized with Python's pickle module for seamless deployment in production environments. Comprehensive
logging ensures traceability, making the pipeline transparent and adaptable. This project highlights the indispensable role of fraud detection in safeguarding financial integrity and demonstrates an end-to-end solution that integrates data engineering, advanced
feature engineering, and state-of-the-art machine learning algorithms to build scalable, reliable, and high-performing predictive systems.

<h2 align = "left">Dataset</h2>

ineuron.ai

<h2 align = "left">Installation</h2>

- Python
- Jupyter, VS Code
- Mongo DB
- Airflow
- Docker
- Fast API

<h2 align = "left">Setup</h2>

- Create appropriate pyenv environment
- Install required dependencies
- Execute and get the pkl file
- Feed the pkl top the api for prediction

<h2 align = "left">Demo</h2>
