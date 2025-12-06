from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import sys

# Add parent directory to path to import scripts
sys.path.insert(0, '/opt/airflow')

# Import your scripts
from scripts.etl_logic import process_new_data
from scripts.scraping_logic import update_poi_data  # <--- NEW IMPORT

def run_scraping_task():
    poi_path = '/opt/airflow/data/poi_bangkok.csv'
    update_poi_data(poi_path)

def run_etl_task():
    # Define paths
    new_data_path = '/opt/airflow/data/new_incoming_data.csv'
    master_path = '/opt/airflow/data/master_dataset.parquet'
    poi_path = '/opt/airflow/data/poi_bangkok.csv'
    
    # Check if new data exists before running
    if not os.path.exists(new_data_path):
        print("No new data file found. Ending task.")
        return

    # Run the main logic
    process_new_data(new_data_path, master_path, poi_path)

# Define DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('bangkok_complaints_etl', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    
    # Task 1: Scrape POI data (only if old/missing)
    scrape_task = PythonOperator(
        task_id='fetch_poi_data',
        python_callable=run_scraping_task
    )

    # Task 2: Process Complaints (Uses the POI data)
    process_task = PythonOperator(
        task_id='process_traffy_data',
        python_callable=run_etl_task
    )

    # Set Dependency: Scrape MUST finish before Processing starts
    scrape_task >> process_task