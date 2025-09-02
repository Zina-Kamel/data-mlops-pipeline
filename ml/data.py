import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import boto3
import torch
from torch.utils.data import Dataset
from config import Config

def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=Config.S3_ENDPOINT,
        aws_access_key_id=Config.ACCESS_KEY,
        aws_secret_access_key=Config.SECRET_KEY,
    )

def load_episode_from_s3(key: str) -> pd.DataFrame:
    """Load a parquet episode file from S3 into a DataFrame."""
    s3 = _get_s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(Config.S3_BUCKET, key, buf)
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pandas()

def flatten_episode(df: pd.DataFrame) -> np.ndarray:
    """Flatten episode dataframe into 1D numpy."""
    obs_cols = ["obs.ee_pose", "obs.ee_quat", "obs.joint_pose",
                "obs.joint_vel", "obs.gripper_width"]
    for col in obs_cols:
        df[col] = df[col].apply(
            lambda x: np.array(x, dtype=np.float32).flatten() if x is not None else np.zeros(1, dtype=np.float32)
        )
    return np.concatenate(df[obs_cols].apply(lambda row: np.concatenate(row.values), axis=1).tolist(), axis=0)

def load_preferences() -> list[dict]:
    """Load preference JSONs from S3 bucket."""
    s3 = _get_s3_client()
    response = s3.list_objects_v2(Bucket=Config.S3_BUCKET, Prefix="rlhf_preferences/")
    keys = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.json')]

    prefs = []
    for k in keys[:-1]:  
        buf = io.BytesIO()
        s3.download_fileobj(Config.S3_BUCKET, k, buf)
        buf.seek(0)
        data = json.load(buf)

        if isinstance(data, list):
            prefs.extend([d for d in data if isinstance(d, dict)])
        elif isinstance(data, dict):
            prefs.append(data)
        else:
            raise ValueError(f"Unexpected JSON structure in {k}: {data}")

    return prefs

class PreferenceDataset(Dataset):
    """Dataset for preference learning."""

    def __init__(self, preferences: list[dict], pad_to_length: int | None = None):
        self.data = preferences
        self.pad_to_length = pad_to_length  

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        pref = self.data[idx]
        x_a = torch.tensor(flatten_episode(load_episode_from_s3(pref["episode_a"])), dtype=torch.float32)
        x_b = torch.tensor(flatten_episode(load_episode_from_s3(pref["episode_b"])), dtype=torch.float32)
        y = torch.tensor([1.0 if pref["preferred"] == pref["episode_a"] else 0.0], dtype=torch.float32)

        if self.pad_to_length:
            x_a = torch.cat([x_a, torch.zeros(self.pad_to_length - x_a.numel())])[:self.pad_to_length]
            x_b = torch.cat([x_b, torch.zeros(self.pad_to_length - x_b.numel())])[:self.pad_to_length]

        return x_a, x_b, y
