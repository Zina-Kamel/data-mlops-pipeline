import mlflow
from config import Config, seed_everything
from train import train_models

def main():
    seed_everything()
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    best_model = train_models()
    print(f"Best model: {best_model}")

if __name__ == "__main__":
    main()
