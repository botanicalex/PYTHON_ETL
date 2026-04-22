import io
import json
import os
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_ROOT, "config.json")

with open(_CONFIG_PATH, "r") as f:
    _config = json.load(f)

BUCKET_NAME = _config["bucket"]
GCP_PROJECT = _config["gcp_project"]
CREDENTIALS_PATH = os.path.join(_ROOT, _config["credentials_path"])
DATA_FILE = _config["data_file"]


def load_data_from_gcp(
    bucket_name: str = BUCKET_NAME,
    blob_name: str = DATA_FILE,
    credentials_path: str = CREDENTIALS_PATH,
) -> pd.DataFrame:
    """Descarga un archivo Excel desde GCP Storage y retorna un DataFrame."""
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = storage.Client(project=GCP_PROJECT, credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    data = blob.download_as_bytes()
    df = pd.read_excel(io.BytesIO(data))
    print(f"Datos descargados desde gs://{bucket_name}/{blob_name} — {df.shape[0]} filas x {df.shape[1]} columnas")
    return df
