from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import boto3
import io
import os
from dotenv import load_dotenv

load_dotenv()

S3_PREFIX = "rlhf_preferences/"
S3_ENDPOINT = os.getenv("MINIO_ENDPOINT")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
S3_BUCKET = os.getenv("BUCKET")

TRAINED_FILES_KEY = "trained_files/trained_files.txt"

def check_new_data_ready(**kwargs):
    """
    Check if at least 10 new preference files have been added to S3 since the last training.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    all_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")]

    trained_files = []
    try:
        trained_obj = s3.get_object(Bucket=S3_BUCKET, Key=TRAINED_FILES_KEY)
        trained_files = trained_obj["Body"].read().decode("utf-8").splitlines()
    except s3.exceptions.NoSuchKey:
        print("Trained files list does not exist yet. Creating new one.")

    new_files = list(set(all_files) - set(trained_files))
    print(f"Found {len(new_files)} new preference files")

    if len(new_files) >= 10:
        updated_content = "\n".join(trained_files + new_files) if trained_files else "\n".join(new_files)

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=TRAINED_FILES_KEY,
            Body=updated_content.encode("utf-8"),
        )
        print(f"Updated trained files list with {len(new_files)} new files.")
        return True

    raise ValueError("Not enough new episodes to trigger training")

with DAG(
    dag_id="conditional_training_new_episodes_s3",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
) as dag:

    check_data = PythonOperator(
        task_id="check_new_data_ready",
        python_callable=check_new_data_ready,
    )

    run_training = DockerOperator(
        task_id="run_training",
        image="ml-train:latest",        
        command="python /app/train.py", 
        docker_url="unix://var/run/docker.sock",
        network_mode='data-mlops-pipeline_default',         
        mount_tmp_dir=False,   
        force_pull=False, 
        environment={
            "MLFLOW_TRACKING_URI": "http://data-mlops-pipeline-mlflow-1:5000",
            "AWS_ACCESS_KEY_ID": ACCESS_KEY,
            "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000"
        },
    )

    check_data >> run_training
