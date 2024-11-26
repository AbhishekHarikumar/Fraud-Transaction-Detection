from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline import MachineLearningModel
from datetime import datetime


def run_ml_pipeline():
    """Run the machine learning pipeline for fraud detection."""
    ml_pipeline = MachineLearningModel()
    ml_pipeline.best_model_selected()


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 25),
    'retries': 1,
}

with DAG(
    'fraud_detection_ml_pipeline',
    default_args=default_args,
    description='ML pipeline for fraud detection',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    ml_task = PythonOperator(
        task_id='run_ml_pipeline_task',
        python_callable=run_ml_pipeline,
    )

    ml_task
