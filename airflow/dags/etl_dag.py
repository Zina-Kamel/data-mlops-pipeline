from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import io
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET = os.getenv("BUCKET")

SOURCE_PREFIX = "etl-output/accepted_episodes/"
TARGET_PREFIX = "etl-output/dashboard/"

def list_s3_keys(prefix=SOURCE_PREFIX):
    s3 = boto3.client("s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [item["Key"] for item in response.get("Contents", []) if item["Key"].endswith(".parquet")]

def transform_episode(df: pd.DataFrame) -> pd.DataFrame:
    df_flat = pd.json_normalize(df.to_dict(orient="records"))

    if "`obs.gripper_width`" in df_flat.columns:
        df_flat["gripper_open"] = df_flat["`obs.gripper_width`"] > 0.05

    return df_flat

def etl_task():
    s3 = boto3.client("s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    keys = list_s3_keys()
    all_dfs = []

    for key in keys:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, key, buf)
        buf.seek(0)
        table = pq.read_table(buf)
        df = table.to_pandas()
        df_flat = transform_episode(df)
        all_dfs.append(df_flat)

    if not all_dfs:
        print("No accepted episodes found to process")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    buffer_out = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(final_df), buffer_out)
    buffer_out.seek(0)
    target_key = f"{TARGET_PREFIX}dashboard.parquet"
    s3.upload_fileobj(buffer_out, BUCKET, target_key)
    print(f"Dashboard Parquet updated at {target_key}")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    "etl_dashboard_accepted",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@hourly",  
    catchup=False,
    default_args=default_args
) as dag:

    run_etl = PythonOperator(
        task_id="extract_transform_load",
        python_callable=etl_task
    )
