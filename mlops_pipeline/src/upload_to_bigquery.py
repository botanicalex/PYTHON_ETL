import io
import json
import os
from google.cloud import storage, bigquery
from google.oauth2 import service_account
import pandas as pd

_ROOT            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREDENTIALS_PATH = os.path.join(_ROOT, "gcp_credentials.json")
GCP_PROJECT      = "credit-risk-mlops-alex"
BUCKET_NAME      = "credit-risk-mlops-alex-data"
DATA_FILE        = "Base_de_datos.xlsx"
BQ_DATASET       = "credit_risk_dataset_east"
BQ_DATASET_REGION = "us-east1"
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

def create_dataset_if_not_exists(client: bigquery.Client):
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
    dataset_ref.location = BQ_DATASET_REGION
    dataset = client.create_dataset(dataset_ref, exists_ok=True)
    print(f"Dataset listo: {dataset.dataset_id} (region: {dataset.location})")

def upload_to_bigquery(df: pd.DataFrame, credentials) -> int:
    client = bigquery.Client(project=GCP_PROJECT, credentials=credentials)
    create_dataset_if_not_exists(client)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    job = client.load_table_from_file(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
        table_id,
        job_config=job_config,
    )
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

    print(f"\nCarga completada: {filas} filas subidas a {GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}")
