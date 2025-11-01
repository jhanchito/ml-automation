from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model_utils import save_model_bundle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import os
import joblib
import numpy as np

# def train_and_eval():
#     iris = load_iris(as_frame=True)
#     X, y = iris.data, iris.target
#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
#     yhat = clf.predict(Xte)
#     metrics = {
#         "accuracy": float(accuracy_score(yte, yhat)),
#         "f1_macro": float(f1_score(yte, yhat, average="macro")),
#         "classes": list(iris.target_names)
#     }
#     bundle = {"model": clf, "target_names": iris.target_names.tolist()}
#     save_model_bundle(bundle)
#     print("Train OK:", metrics)
#     return metrics
    
def train_and_save():
    # Cargar el dataset desde el archivo CSV
    try:
        df = pd.read_csv("churn-bigml-80.csv")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'churn-bigml-80.csv'")
        exit()

    numeric_features = [
    "Account length", "Number vmail messages", "Total day minutes",
    "Total day calls", "Total day charge", "Total eve minutes",
    "Total eve calls", "Total eve charge", "Total night minutes",
    "Total night calls", "Total night charge", "Total intl minutes",
    "Total intl calls", "Total intl charge", "Customer service calls"
    ]

    X = df[numeric_features]

    # Convertir la columna objetivo 'Churn' (que es True/False) a números (1/0)
    # Esto es necesario para que el modelo de regresión logística funcione
    y = df["Churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, multi_class="auto", random_state=8888)
    # pipe = Pipeline(steps=[
    #     ("scaler", StandardScaler()),
    #     ("clf", LogisticRegression(max_iter=1000, multi_class="auto", random_state=8888))
    # ])
    yhat = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_train)),
        "f1_macro": float(f1_score(y_test, y_train, average="macro")),
        "classes": df["Churn"].unique().tolist()
    }
    bundle = {"model": clf, "target_names": iris.target_names.tolist()}
    save_model_bundle(bundle)
    print("Train OK:", metrics)
    return metrics


    pipe.fit(X_train, y_train)

    # Evaluación
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "target_names": df["Churn"].unique(),
        "feature_names": numeric_features
    }, model_path)
    print(f"Modelo guardado en: {model_path}")

if __name__ == "__main__":
    train_and_eval()
