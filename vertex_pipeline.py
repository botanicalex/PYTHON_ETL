import os
from google.cloud import aiplatform
from google.oauth2 import service_account
from kfp import dsl, compiler
from kfp.dsl import component, Dataset, Model, Metrics, Output, Input

GCP_PROJECT      = "credit-risk-mlops-alex"
GCP_REGION       = "us-east1"
BUCKET           = "credit-risk-mlops-alex-data"
BQ_TABLE         = "credit-risk-mlops-alex.credit_risk_dataset_east.creditos_raw"
TARGET           = "Pago_atiempo"
PIPELINE_NAME    = "credit-risk-pipeline"
CREDENTIALS_PATH = "gcp_credentials.json"
SERVICE_ACCOUNT  = "credit-risk-sa@credit-risk-mlops-alex.iam.gserviceaccount.com"

COLS_DROP = [
    "cant_creditosvigentes", "cuota_pactada", "fecha_prestamo",
    "puntaje", "promedio_ingresos_datacredito", "tendencia_ingresos", "saldo_principal",
]


# ─────────────────────────────────────────────
# Componente 1 — Cargar datos desde BigQuery
# ─────────────────────────────────────────────
@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"],
)
def cargar_datos(
    project: str,
    bq_table: str,
    output_csv: Output[Dataset],
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project)
    df = client.query(f"SELECT * FROM `{bq_table}`").to_dataframe()
    print(f"Filas cargadas desde BigQuery: {len(df)}")
    df.to_csv(output_csv.path, index=False)


# ─────────────────────────────────────────────
# Componente 2 — Preprocesar
# ─────────────────────────────────────────────
@component(
    base_image="python:3.11",
    packages_to_install=["pandas"],
)
def preprocesar(
    input_csv: Input[Dataset],
    cols_drop: list,
    output_csv: Output[Dataset],
):
    import pandas as pd

    df = pd.read_csv(input_csv.path)
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    df = df.dropna()
    print(f"Dataset limpio: {df.shape[0]} filas x {df.shape[1]} columnas")
    df.to_csv(output_csv.path, index=False)


# ─────────────────────────────────────────────
# Componente 3 — Entrenar LogisticRegression
# ─────────────────────────────────────────────
@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib", "google-cloud-storage"],
)
def entrenar(
    input_csv: Input[Dataset],
    target: str,
    bucket: str,
    output_model: Output[Model],
    metrics: Output[Metrics],
):
    import io
    import pandas as pd
    import joblib
    from google.cloud import storage
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(input_csv.path)
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols     = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = round(accuracy_score(y_test, y_pred), 4)
    f1      = round(f1_score(y_test, y_pred, zero_division=0), 4)
    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    recall  = round(recall_score(y_test, y_pred, zero_division=0), 4)

    metrics.log_metric("accuracy", acc)
    metrics.log_metric("f1_score", f1)
    metrics.log_metric("roc_auc",  roc_auc)
    metrics.log_metric("recall",   recall)

    print(f"Accuracy: {acc} | F1: {f1} | ROC-AUC: {roc_auc} | Recall: {recall}")

    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    gcs_client = storage.Client()
    bucket_obj = gcs_client.bucket(bucket)
    bucket_obj.blob("models/logistic_regression_vertex.pkl").upload_from_file(
        model_bytes, content_type="application/octet-stream"
    )
    print(f"Modelo guardado en gs://{bucket}/models/logistic_regression_vertex.pkl")

    joblib.dump(model, output_model.path)


# ─────────────────────────────────────────────
# Componente 4 — Evaluar y registrar en Vertex AI Experiments
# ─────────────────────────────────────────────
@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"],
)
def evaluar(
    project: str,
    region: str,
    experiment_name: str,
    run_name: str,
    input_metrics: Input[Metrics],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, experiment=experiment_name)

    with aiplatform.start_run(run_name):
        aiplatform.log_metrics({k: v for k, v in input_metrics.metadata.items()})

    print(f"Metricas registradas en experimento '{experiment_name}', run '{run_name}'")


# ─────────────────────────────────────────────
# Definicion del pipeline
# ─────────────────────────────────────────────
@dsl.pipeline(
    name="credit-risk-pipeline",
    description="Carga, preprocesamiento, entrenamiento y evaluacion de riesgo crediticio",
)
def credit_risk_pipeline(
    project: str        = GCP_PROJECT,
    region: str         = GCP_REGION,
    bucket: str         = BUCKET,
    bq_table: str       = BQ_TABLE,
    target: str         = TARGET,
    cols_drop: list     = COLS_DROP,
    experiment_name: str = "credit-risk-experiment",
    run_name: str        = "run-logistic-v1",
):
    step1 = cargar_datos(project=project, bq_table=bq_table)

    step2 = preprocesar(
        input_csv=step1.outputs["output_csv"],
        cols_drop=cols_drop,
    )

    step3 = entrenar(
        input_csv=step2.outputs["output_csv"],
        target=target,
        bucket=bucket,
    )

    evaluar(
        project=project,
        region=region,
        experiment_name=experiment_name,
        run_name=run_name,
        input_metrics=step3.outputs["metrics"],
    )


# ─────────────────────────────────────────────
# Compilar y enviar a Vertex AI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    PIPELINE_FILE = "credit_risk_pipeline.json"

    print("=" * 55)
    print("   VERTEX AI PIPELINE — vertex_pipeline.py")
    print("=" * 55)

    print("\nCompilando pipeline...")
    compiler.Compiler().compile(pipeline_func=credit_risk_pipeline, package_path=PIPELINE_FILE)
    print(f"Pipeline compilado: {PIPELINE_FILE}")

    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        credentials=credentials,
    )

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=PIPELINE_FILE,
        pipeline_root=f"gs://{BUCKET}/pipeline_root",
        enable_caching=False,
    )

    print("\nEnviando pipeline a Vertex AI...")
    job.submit()
    print(f"\nPipeline enviado.")
    print(f"Nombre : {job.name}")
    print(f"Console: https://console.cloud.google.com/vertex-ai/pipelines?project={GCP_PROJECT}")
