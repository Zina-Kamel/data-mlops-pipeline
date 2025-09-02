import os
import random
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    # S3 + MinIO
    S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    S3_BUCKET = os.getenv("MINIO_BUCKET", "robot-data")
    ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

    # MLflow
    MLFLOW_URI = os.getenv("MLFLOW_URI", "http://data-mlops-pipeline-mlflow-1:5000")
    EXPERIMENT_NAME = "rlhf_reward_models"

    # Training
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 7
    LR = 1e-3
    PAD_LENGTH = 250
    ARTIFACT_DIR = "artifacts"

def seed_everything(seed: int = Config.SEED):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
