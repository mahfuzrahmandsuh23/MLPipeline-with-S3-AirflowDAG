from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tasks
from src.preprocessing import preprocess_main
from src.train_model import train_main
from src.evaluate_model import evaluate_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 1),
    'retries': 1,
}

with DAG(
    dag_id='carbon_emission_ml_pipeline',
    default_args=default_args,
    schedule='@daily',  # ✅ Airflow 3.x
    catchup=False
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_main
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_main
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    preprocess_task >> train_task >> evaluate_task
