from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, '/opt/airflow')

# --- TASKS ---
def run_scraping_task():
    # Lazy load to prevent import errors
    from scripts.scraping_logic import update_poi_data
    
    print("--- Starting POI Scraping ---")
    poi_path = '/opt/airflow/data/poi_bangkok.csv'
    
    # Run the logic
    update_poi_data(poi_path)
    print("--- POI Scraping Completed ---")

def run_etl_task():
    from scripts.etl_logic import process_new_data
    
    print("--- Starting ETL Process ---")
    new_data_path = '/opt/airflow/data/new_incoming_data.csv'
    master_path = '/opt/airflow/data/master_dataset.parquet'
    poi_path = '/opt/airflow/data/poi_bangkok.csv'
    
    if not os.path.exists(new_data_path):
        print(f"No file found at {new_data_path}. Skipping.")
        return

    process_new_data(new_data_path, master_path, poi_path)
    print("--- ETL Process Completed ---")

# --- DAG DEFINITION ---

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # SAFETY NET: If a task runs longer than 60 mins, kill it.
    'execution_timeout': timedelta(minutes=60), 
}

with DAG(
    'bangkok_complaints_etl', 
    default_args=default_args, 
    schedule_interval='@daily', 
    catchup=False
) as dag:
    
    scrape_task = PythonOperator(
        task_id='fetch_poi_data',
        python_callable=run_scraping_task,
        # You can also set specific timeouts per task
        execution_timeout=timedelta(minutes=30), 
    )

    process_task = PythonOperator(
        task_id='process_traffy_data',
        python_callable=run_etl_task,
        execution_timeout=timedelta(minutes=120),
    )

    scrape_task >> process_task