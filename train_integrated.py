import os
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    data_path = "data/train.csv"

    # Create dataset if not present
    if not os.path.exists(data_path):
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "feature": [0.5, 1.2, 2.1, 1.8], "label": [1, 0, 1, 0]}
        )
        df.to_csv(data_path, index=False)

    # Get DVC data hash / version via subprocess if initialized, or fallback
    try:
        dvc_version = (
            subprocess.check_output(["dvc", "hash", data_path]).decode().strip()
        )
    except Exception:
        dvc_version = "v1.0-local"

    mlflow.set_experiment("DVC_MLflow_Integrated_Project")

    with mlflow.start_run():
        mlflow.log_param("dataset_version", dvc_version)
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)

        # Simple training logic
        X = [[0.5], [1.2], [2.1], [1.8]]
        y = [1, 0, 1, 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        # Tree is trusted: we trained this model ourselves
        mlflow.sklearn.log_model(
            model,
            name="integrated_rf_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("Integrated DVC + MLflow experiment completed and logged successfully!")
        print(f"Tracked DVC dataset version: {dvc_version}")
        print(f"Accuracy: {accuracy}")
        print("Run `mlflow ui` to view results.")
