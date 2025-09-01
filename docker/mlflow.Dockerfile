FROM ghcr.io/mlflow/mlflow:latest

RUN pip install psycopg2-binary
RUN pip install boto3