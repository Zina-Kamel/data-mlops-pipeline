# Franka arm data collection pipeline with RLHF recommendation system 

![MLflow](https://img.shields.io/badge/MLflow-blue?logo=mlflow)  ![Airflow](https://img.shields.io/badge/Airflow-orange?logo=apacheairflow)  ![Kafka](https://img.shields.io/badge/Kafka-black?logo=apachekafka)  ![Docker](https://img.shields.io/badge/Docker-blue?logo=docker)  ![Spark](https://img.shields.io/badge/Spark-orange?logo=apachespark)  ![Redis](https://img.shields.io/badge/Redis-red?logo=redis)  ![Postgres](https://img.shields.io/badge/Postgres-blue?logo=postgresql)  ![MinIO](https://img.shields.io/badge/MinIO-green?logo=minio)  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)


---

## Overview  

This project aims to improve the traditionally manual and uninteractive process of data collection and visualization when collecting 
data from Franka arm. It offers end-to-end pipeline to collect and visualise robotic 
trajectories. Users can visualise the collected trajectory and its statistics then choose to accept or reject it. Powered
by RLHF fine tuned reward model, the pipeline also includes a recommendation whether the collected trajectory aligns 
with human preference.


## Features  
- **Data ingestion and streaming** using **Apache Kafka** and **Redis**. 
- **ETL and batch processing pipeline** orchestrated with **Apache Airflow**, with **Apache Spark** for scalable data transformations and computation of aggregate statistics.
- **Training, evaluating and comparing models** with **MLflow**.  
- **Storage** with **MinIO** (object store) and **PostgreSQL** (metadata DB).   
- **Orchestration** and **Containerization** with **Docker** and **Airflow**.  
- **Continual retraining** to adapt models over time with new data.  
- **Interactive visualization** with a **Streamlit dashboard**.  

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Zina-Kamel/data-mlops-pipeline.git
cd data-mlops-pipeline
```

### 2. Create .env file

Create a .env file at the root of the project and populate the following variables:

```bash
AIRFLOW_ADMIN_USER=
AIRFLOW_ADMIN_PASSWORD=
AIRFLOW__WEBSERVER__SECRET_KEY=

MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
BUCKET= # name of the main bucket for the stored trajectories

POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
```

### 3. Build and start services

```bash
docker compose up --build
```

This will start the following services:

- **Kafka and Zookeeper** (streaming)  
- **Redis** (cache)  
- **Producer and Consumer** (data pipelines)  
- **MinIO** (object storage)  
- **PostgreSQL** (metadata)  
- **Airflow Webserver and Scheduler** (ETL and retraining orchestration)  
- **MLflow** (model tracking)  
- **Streamlit Dashboard**

This will automatically start the Kafka producer to stream data to the consumer and then to the Streamlit application.

> **Note:** If you have already collected trajectories that you want to visualize or filter, you can upload the HDF5 file under `sample_data/`.

The current RLHF reward model is trained on a variety of tasks, but you can retrain it on your own data by:  
1. Accepting trajectories in the Live tab.  
2. Ranking trajectories in the RLHF tab.  

Once the new data is added using the RLHF tab, Airflow automatically triggers training on the updated dataset.