FROM apache/airflow:2.8.1

USER airflow

RUN pip install --no-cache-dir apache-airflow-providers-docker
RUN pip install python-dotenv

USER root

RUN groupadd -g 125 docker && usermod -aG docker airflow

USER airflow
