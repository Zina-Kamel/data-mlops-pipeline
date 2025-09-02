import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from config import Config
from utils import collate_fn_mlp
from data import PreferenceDataset, load_preferences
from models import MLP, TransformerModel

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()
    for x_a, x_b, y in dataloader:
        x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
        r_a, r_b = model(x_a), model(x_b)
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
            r_a, r_b = model(x_a), model(x_b)
            prob = torch.sigmoid(r_a - r_b)
            loss = criterion(prob, y)
            total_loss += loss.item() * x_a.size(0)
    return total_loss / len(dataloader.dataset)

def train_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preferences = load_preferences()
    split = int(0.8 * len(preferences))
    train_ds = PreferenceDataset(preferences[:split], pad_to_length=Config.PAD_LENGTH)
    val_ds = PreferenceDataset(preferences[split:], pad_to_length=Config.PAD_LENGTH)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                           shuffle=True, collate_fn=collate_fn_mlp)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                                         shuffle=False, collate_fn=collate_fn_mlp)

    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)

    models = {"MLP": MLP, "Transformer": TransformerModel}
    best_model = None

    for name, ModelClass in models.items():
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_type", name)
            model = ModelClass(input_size=Config.PAD_LENGTH, output_size=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=Config.LR)

            best_val_loss = float("inf")
            epochs_no_improve = 0

            for epoch in range(Config.EPOCHS):
                train_loss = train_epoch(model, train_dl, optimizer, device)
                val_loss = evaluate(model, val_dl, device)
                mlflow.log_metric(f"{name}_train_loss", train_loss, step=epoch)
                mlflow.log_metric(f"{name}_val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= Config.PATIENCE:
                        break

            model.load_state_dict(best_weights)
            model_path = os.path.join(Config.ARTIFACT_DIR, f"{name}_reward_model.pth")
            torch.save(model.state_dict(), model_path)
            mlflow.pytorch.log_model(model, artifact_path=f"{name}_model")

            if best_model is None or best_val_loss < best_model["val_loss"]:
                best_model = {"model_type": name, "val_loss": best_val_loss,
                              "model_path": os.path.basename(model_path)}

    best_model_json_path = os.path.join(Config.ARTIFACT_DIR, "best_model.json")
    with open(best_model_json_path, "w") as f:
        json.dump(best_model, f, indent=4)

    return best_model
