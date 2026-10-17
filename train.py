import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load sample dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run() as run:
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and model artifact
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Model Accuracy: {acc}")
