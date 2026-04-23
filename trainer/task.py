import io
import json
import os
import joblib
from google.cloud import bigquery, storage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

GCP_PROJECT  = "credit-risk-mlops-alex"
BQ_TABLE     = "credit-risk-mlops-alex.credit_risk_dataset_east.creditos_raw"
BUCKET       = "credit-risk-mlops-alex-data"
GCS_PREFIX   = "custom-training"
TARGET       = "Pago_atiempo"

COLS_DROP = [
    "cant_creditosvigentes", "cuota_pactada", "fecha_prestamo",
    "puntaje", "promedio_ingresos_datacredito", "tendencia_ingresos", "saldo_principal",
]


def load_data() -> pd.DataFrame:
    print("Cargando datos desde BigQuery...")
    client = bigquery.Client(project=GCP_PROJECT)
    df = client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()
    print(f"Filas cargadas: {len(df)}")
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    df = df.drop(columns=[c for c in COLS_DROP if c in df.columns])
    df = df.dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print(f"Dataset preprocesado: {X.shape[0]} filas x {X.shape[1]} features")
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols     = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def save_to_gcs(data: bytes, blob_name: str, content_type: str = "application/octet-stream"):
    client = storage.Client(project=GCP_PROJECT)
    blob = client.bucket(BUCKET).blob(blob_name)
    blob.upload_from_file(io.BytesIO(data), content_type=content_type)
    print(f"Guardado en gs://{BUCKET}/{blob_name}")


def main():
    print("=" * 55)
    print("   CUSTOM TRAINING JOB — trainer/task.py")
    print("=" * 55)

    # 1. Cargar datos
    df = load_data()

    # 2. Preprocesar
    X, y = preprocess(df)

    # 3. Split y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Entrenando LogisticRegression con class_weight=balanced...")
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"Metricas: {metrics}")

    # 4. Guardar metrics.json en GCS
    metrics_bytes = json.dumps(metrics, indent=2).encode("utf-8")
    save_to_gcs(metrics_bytes, f"{GCS_PREFIX}/metrics.json", content_type="application/json")

    # 5. Guardar modelo en GCS
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    save_to_gcs(model_buffer.read(), f"{GCS_PREFIX}/model.pkl")

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
