FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    numpy \
    matplotlib \
    seaborn \
    boto3 \
    torch

COPY ml/train.py .

CMD ["python", "train.py"]
