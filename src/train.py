# src/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from src.config import *

def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = pd.read_csv(TRAIN_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]
    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
    }

    with mlflow.start_run():
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "recall": recall_score(y_test, y_pred),
        }

        mlflow.log_params({k: v for k, v in params.items() if k != "use_label_encoder"})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud-detector",
        )

        print(f"Logged metrics: {metrics}")
        print("Registered model 'fraud-detector' created.")

if __name__ == "__main__":
    train_model()