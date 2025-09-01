import mlflow
import boto3
import pandas as pd
import pyarrow.parquet as pq
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
from torch.nn.utils.rnn import pad_sequence
import boto3

from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

S3_ENDPOINT = "http://minio:9000"
S3_BUCKET = "robot-data"
EPISODE_PREFIX = "etl-output/rlhf_episodes/"
PREFERENCE_PREFIX = "rlhf_preferences/"

AWS_ACCESS_KEY = ACCESS_KEY
AWS_SECRET_KEY = SECRET_KEY

mlflow.set_tracking_uri("http://data-mlops-pipeline-mlflow-1:5000")
mlflow.set_experiment("rlhf_reward_models")

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 7
LR = 1e-3

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

class MLP(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_size=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.fc_out(x)

def load_episode_from_s3(key):
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT,
                      aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY)
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    table = pq.read_table(buf)
    df = table.to_pandas()
    return df

def flatten_episode(df):
    obs_cols = ["obs.ee_pose", "obs.ee_quat", "obs.joint_pose",
                "obs.joint_vel", "obs.gripper_width"]
    for col in obs_cols:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float32).flatten() if x is not None else np.zeros(1, dtype=np.float32))
    return np.concatenate(df[obs_cols].apply(lambda row: np.concatenate(row.values), axis=1).tolist(), axis=0)

def load_preferences():
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT,
                      aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY)
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=PREFERENCE_PREFIX)
    keys = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.json')]
    
    prefs = []
    for k in keys[:len(keys)-1]:
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, k, buf)
        buf.seek(0)
        data = json.load(buf)
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    prefs.append(d)
                else:
                    raise ValueError(f"Unexpected non-dict item in preference JSON: {d}")
        elif isinstance(data, dict):
            prefs.append(data)
        else:
            raise ValueError(f"Unexpected JSON structure in {k}: {data}")

    return prefs

class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, preferences, pad_to_length=None):
        self.data = preferences
        self.pad_to_length = pad_to_length  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pref = self.data[idx]
        x_a = torch.tensor(flatten_episode(load_episode_from_s3(pref["episode_a"])), dtype=torch.float32)
        x_b = torch.tensor(flatten_episode(load_episode_from_s3(pref["episode_b"])), dtype=torch.float32)
        y = torch.tensor([1.0 if pref["preferred"] == pref["episode_a"] else 0.0], dtype=torch.float32)

        if self.pad_to_length is not None:
            if x_a.numel() < self.pad_to_length:
                x_a = torch.cat([x_a, torch.zeros(self.pad_to_length - x_a.numel())])
            else:
                x_a = x_a[:self.pad_to_length]

            if x_b.numel() < self.pad_to_length:
                x_b = torch.cat([x_b, torch.zeros(self.pad_to_length - x_b.numel())])
            else:
                x_b = x_b[:self.pad_to_length]

        return x_a, x_b, y

def collate_fn_mlp(batch):
    x_a_list, x_b_list, y_list = zip(*batch)
    x_a_tensor = torch.stack(x_a_list)  # [batch_size, pad_to_length]
    x_b_tensor = torch.stack(x_b_list)
    y_tensor = torch.stack(y_list)
    return x_a_tensor, x_b_tensor, y_tensor

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()
    for x_a, x_b, y in dataloader:
        x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
        r_a = model(x_a)
        r_b = model(x_b)
        prob = torch.sigmoid(r_a - r_b)
        loss = criterion(prob, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_a.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for x_a, x_b, y in dataloader:
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            r_a = model(x_a)
            r_b = model(x_b)
            prob = torch.sigmoid(r_a - r_b)
            loss = criterion(prob, y)
            total_loss += loss.item() * x_a.size(0)
    return total_loss / len(dataloader.dataset)

def collate_fn(batch):
    x_a_list, x_b_list, y_list = zip(*batch)
    x_a_padded = pad_sequence(x_a_list, batch_first=True, padding_value=0)
    x_b_padded = pad_sequence(x_b_list, batch_first=True, padding_value=0)
    y_tensor = torch.stack(y_list)
    return x_a_padded, x_b_padded, y_tensor

def train_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preferences = load_preferences()
    random.shuffle(preferences)
    split = int(0.8 * len(preferences))
    train_data = preferences[:split]
    val_data = preferences[split:]

    PAD_LENGTH = 250  

    train_ds = PreferenceDataset(train_data, pad_to_length=PAD_LENGTH)
    val_ds = PreferenceDataset(val_data, pad_to_length=PAD_LENGTH)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_mlp)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_mlp)

    ARTIFACT_DIR = "artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    models = {"MLP": MLP, "Transformer": TransformerModel}
    best_model = None

    for name, ModelClass in models.items():
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_type", name)
            dummy_input = torch.zeros(PAD_LENGTH) 
            model = ModelClass(input_size=dummy_input.shape[0], output_size=1).to(device)

            optimizer = optim.Adam(model.parameters(), lr=LR)
            best_val_loss = float("inf")
            epochs_no_improve = 0

            for epoch in range(EPOCHS):
                train_loss = train_epoch(model, train_dl, optimizer, device)
                val_loss = evaluate(model, val_dl, device)
                mlflow.log_metric(f"{name}_train_loss", train_loss, step=epoch)
                mlflow.log_metric(f"{name}_val_loss", val_loss, step=epoch)
                print(f"{name} Epoch {epoch+1} - Train: {train_loss:.4f} - Val: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        print(f"Early stopping for {name}")
                        break

            model.load_state_dict(best_weights)
            model_path = os.path.join(ARTIFACT_DIR, f"{name}_reward_model.pth")
            torch.save(model.state_dict(), model_path)
            mlflow.pytorch.log_model(model, artifact_path=f"{name}_model")
            print(f"{name} saved to {model_path}")

            if best_model is None or best_val_loss < best_model["val_loss"]:
                best_model = {
                    "model_type": name,
                    "val_loss": best_val_loss,
                    "model_path": os.path.basename(model_path)
                }

    best_model_json_path = os.path.join(ARTIFACT_DIR, "best_model.json")
    with open(best_model_json_path, "w") as f:
        json.dump(best_model, f, indent=4)
    print(f"Best model info saved to {best_model_json_path}")
    print(f"Best model: {best_model['model_type']} with val_loss={best_model['val_loss']:.4f}")

    model_path = os.path.join(ARTIFACT_DIR, f"{name}_reward_model.pth")
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, artifact_path=f"{name}_model")
    print(f"{name} saved to {model_path}")

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    with open(model_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, f"ml/{os.path.basename(model_path)}")
    print(f"{name} model weights uploaded to S3 at ml/{os.path.basename(model_path)}")

    best_model_json_path = os.path.join(ARTIFACT_DIR, "best_model.json")
    with open(best_model_json_path, "w") as f:
        json.dump(best_model, f, indent=4)
    print(f"Best model info saved to {best_model_json_path}")

    with open(best_model_json_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, "ml/best_model.json")
    print("Best model JSON uploaded to S3 at ml/best_model.json")


if __name__ == "__main__":
    train_models()
