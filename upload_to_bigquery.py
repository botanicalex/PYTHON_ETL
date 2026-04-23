import io
import json
from google.cloud import storage, bigquery
from google.oauth2 import service_account
import pandas as pd

CREDENTIALS_PATH = "gcp_credentials.json"
GCP_PROJECT      = "credit-risk-mlops-alex"
BUCKET_NAME      = "credit-risk-mlops-alex-data"
DATA_FILE        = "Base_de_datos.xlsx"
BQ_DATASET       = "credit_risk_dataset"
BQ_TABLE         = "creditos_raw"

def get_credentials():
    return service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

def download_from_gcs(credentials) -> pd.DataFrame:
    client = storage.Client(project=GCP_PROJECT, credentials=credentials)
    blob = client.bucket(BUCKET_NAME).blob(DATA_FILE)
    data = blob.download_as_bytes()
    df = pd.read_excel(io.BytesIO(data))
    print(f"Datos descargados desde gs://{BUCKET_NAME}/{DATA_FILE} — {len(df)} filas x {len(df.columns)} columnas")
    return df

def upload_to_bigquery(df: pd.DataFrame, credentials) -> int:
    client = bigquery.Client(project=GCP_PROJECT, credentials=credentials)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    tabla = client.get_table(table_id)
    return tabla.num_rows

if __name__ == "__main__":
    print("=" * 55)
    print("   UPLOAD TO BIGQUERY — upload_to_bigquery.py")
    print("=" * 55)

    credentials = get_credentials()

    print("\nDescargando datos desde GCS...")
    df = download_from_gcs(credentials)

    print("\nSubiendo datos a BigQuery...")
    filas = upload_to_bigquery(df, credentials)

    print(f"\n✅ Carga completada: {filas} filas subidas a {GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}")
