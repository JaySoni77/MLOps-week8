import argparse
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

def train_model(data_path):
    # ----- Load Data -----
    data = pd.read_csv(data_path)

    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data['species']

    # ----- Train/Test Split -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # ----- Model Parameters -----
    params = {"max_depth": 5, "random_state": 1}
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    # ----- Evaluation -----
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)

    # ----- Save artifact locally -----
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")

    # ----- MLflow logging -----
    mlflow.set_tracking_uri("http://127.0.0.1:8100")
    mlflow.set_experiment("MLOps Assignment Week 8")

    # build run name automatically
    file_name = os.path.basename(data_path)

    if "poisoned" in file_name:
        poison_level = file_name.split("_")[-1].replace(".csv","")
        run_name = f"poisoned_{poison_level}_percent"
    else:
        run_name = "clean_run"
    
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy_score", accuracy)
        mlflow.set_tag("training_info", "Decision Tree on iris dataset")

        # 🔥 NEW TAGS HERE
        mlflow.set_tag("dataset_used", data_path)

        # auto-detect poison level from filename
        if "poisoned" in data_path:
            poison_percentage = os.path.basename(data_path).split("_")[-1].replace(".csv","")
            mlflow.set_tag("poison_percent", poison_percentage)
        else:
            mlflow.set_tag("poison_percent", "0")

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="iris-classifier-dt"
        )

    print(f"Training complete. Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path)