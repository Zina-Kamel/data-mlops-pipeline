from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2023, 1, 1),
}

with DAG('spark_etl_pipeline', default_args=default_args, schedule_interval='@hourly', catchup=False) as dag:
    run_etl = BashOperator(
        task_id='run_spark_etl',
        bash_command='/opt/bitnami/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 /src/etl/etl_data_to_s3.py'
    )
