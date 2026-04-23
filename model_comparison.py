import os
import sys
import warnings
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
)
from google.cloud import aiplatform
from google.oauth2 import service_account

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuracion
# ─────────────────────────────────────────────
GCP_PROJECT      = "credit-risk-mlops-alex"
GCP_REGION       = "us-east1"
AUTOML_MODEL_NAME = "credit-risk-automl-model"
CREDENTIALS_PATH = "gcp_credentials.json"
MODEL_PATH       = os.path.join("mlops_pipeline", "src", "mejor_modelo.pkl")
SRC_PATH         = os.path.join(os.path.dirname(__file__), "mlops_pipeline", "src")

METRICS_NAMES = ["accuracy", "f1_score", "roc_auc", "recall", "precision"]


# ─────────────────────────────────────────────
# 1. Métricas AutoML desde Vertex AI
# ─────────────────────────────────────────────
def get_automl_metrics(credentials) -> dict:
    print("Conectando a Vertex AI para obtener metricas del modelo AutoML...")
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION, credentials=credentials)

    models = aiplatform.Model.list(
        filter=f'display_name="{AUTOML_MODEL_NAME}"',
        order_by="create_time desc",
    )

    if not models:
        raise ValueError(f"No se encontro el modelo '{AUTOML_MODEL_NAME}' en Vertex AI.")

    model = models[0]
    print(f"Modelo encontrado: {model.display_name} ({model.resource_name})")

    evaluations = model.list_model_evaluations()
    if not evaluations:
        raise ValueError("El modelo no tiene evaluaciones disponibles.")

    eval_metrics = evaluations[0].metrics
    print(f"Evaluacion encontrada: {evaluations[0].resource_name}")

    # Extraer métricas del AutoML — buscar en confidenceMetrics el umbral ~0.5
    roc_auc   = eval_metrics.get("auRoc", None)
    log_loss  = eval_metrics.get("logLoss", None)

    # confidenceMetrics es una lista ordenada por confidence_threshold
    confidence_metrics = eval_metrics.get("confidenceMetrics", [])
    cm_at_50 = None
    if confidence_metrics:
        # Buscar el más cercano a 0.5
        cm_at_50 = min(
            confidence_metrics,
            key=lambda x: abs(x.get("confidenceThreshold", 0) - 0.5)
        )

    precision = cm_at_50.get("precision", None) if cm_at_50 else None
    recall    = cm_at_50.get("recall", None)    if cm_at_50 else None
    f1        = cm_at_50.get("f1Score", None)   if cm_at_50 else None
    accuracy  = cm_at_50.get("accuracy", None)  if cm_at_50 else None

    metrics = {
        "accuracy":  round(accuracy,  4) if accuracy  is not None else None,
        "f1_score":  round(f1,        4) if f1        is not None else None,
        "roc_auc":   round(roc_auc,   4) if roc_auc   is not None else None,
        "recall":    round(recall,    4) if recall     is not None else None,
        "precision": round(precision, 4) if precision  is not None else None,
    }

    print(f"Metricas AutoML extraidas: {metrics}")
    return metrics


# ─────────────────────────────────────────────
# 2. Pipeline de features + split
# ─────────────────────────────────────────────
def get_train_test_data():
    print("\nCargando datos y aplicando pipeline de features...")
    sys.path.insert(0, SRC_PATH)
    from ft_engineering import build_feature_pipeline
    from gcp_utils import load_data_from_gcp

    df = load_data_from_gcp()
    X_train, X_test, y_train, y_test, pipeline_ml = build_feature_pipeline(df)
    return X_train, X_test, y_train, y_test, pipeline_ml


# ─────────────────────────────────────────────
# 3. Métricas del modelo local (mejor_modelo.pkl)
# ─────────────────────────────────────────────
def get_local_model_metrics(X_train, X_test, y_train, y_test, pipeline_ml) -> dict:
    print("\nCalculando metricas del modelo local (mejor_modelo.pkl)...")
    model = joblib.load(MODEL_PATH)

    # Fitear pipeline_ml con X_train y transformar ambos conjuntos
    X_train_t = pipeline_ml.fit_transform(X_train, y_train)
    X_test_t  = pipeline_ml.transform(X_test)

    y_pred = model.predict(X_test_t)
    y_prob = model.predict_proba(X_test_t)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"Metricas modelo local: {metrics}")
    return metrics


# ─────────────────────────────────────────────
# 4a. Tabla comparativa en consola
# ─────────────────────────────────────────────
def print_comparison_table(local_metrics: dict, automl_metrics: dict):
    print("\n" + "=" * 60)
    print(f"  {'METRICA':<15} {'MODELO LOCAL':>15} {'AUTOML':>15}  GANADOR")
    print("=" * 60)
    for metric in METRICS_NAMES:
        local_val  = local_metrics.get(metric)
        automl_val = automl_metrics.get(metric)

        local_str  = f"{local_val:.4f}"  if local_val  is not None else "N/A"
        automl_str = f"{automl_val:.4f}" if automl_val is not None else "N/A"

        if local_val is not None and automl_val is not None:
            winner = "Local" if local_val >= automl_val else "AutoML"
        else:
            winner = "-"

        print(f"  {metric:<15} {local_str:>15} {automl_str:>15}  {winner}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 4b. Gráfico de barras
# ─────────────────────────────────────────────
def plot_bar_comparison(local_metrics: dict, automl_metrics: dict, output_path: str):
    x = np.arange(len(METRICS_NAMES))
    width = 0.35

    local_vals  = [local_metrics.get(m, 0) or 0  for m in METRICS_NAMES]
    automl_vals = [automl_metrics.get(m, 0) or 0 for m in METRICS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, local_vals,  width, label="Modelo Local", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, automl_vals, width, label="AutoML",       color="#FF5722", alpha=0.85)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metrica")
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de Modelos: Local vs AutoML")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in METRICS_NAMES])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico de barras guardado: {output_path}")


# ─────────────────────────────────────────────
# 4c. Gráfico de radar
# ─────────────────────────────────────────────
def plot_radar_comparison(local_metrics: dict, automl_metrics: dict, output_path: str):
    categories = METRICS_NAMES
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    local_vals  = [local_metrics.get(m, 0) or 0  for m in categories] + \
                  [local_metrics.get(categories[0], 0) or 0]
    automl_vals = [automl_metrics.get(m, 0) or 0 for m in categories] + \
                  [automl_metrics.get(categories[0], 0) or 0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, local_vals,  "o-", linewidth=2, color="#2196F3", label="Modelo Local")
    ax.fill(angles, local_vals,  alpha=0.2, color="#2196F3")

    ax.plot(angles, automl_vals, "o-", linewidth=2, color="#FF5722", label="AutoML")
    ax.fill(angles, automl_vals, alpha=0.2, color="#FF5722")

    ax.set_thetagrids(np.degrees(angles[:-1]), [m.replace("_", "\n") for m in categories])
    ax.set_ylim(0, 1)
    ax.set_title("Radar: Modelo Local vs AutoML", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico de radar guardado: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   MODEL COMPARISON — model_comparison.py")
    print("=" * 60)

    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # 1. AutoML metrics
    automl_metrics = get_automl_metrics(credentials)

    # 2 & 3. Local model metrics
    X_train, X_test, y_train, y_test, pipeline_ml = get_train_test_data()
    local_metrics = get_local_model_metrics(X_train, X_test, y_train, y_test, pipeline_ml)

    # 4. Comparacion
    print_comparison_table(local_metrics, automl_metrics)

    plot_bar_comparison(local_metrics, automl_metrics, "comparison_bar.png")
    plot_radar_comparison(local_metrics, automl_metrics, "comparison_radar.png")

    print("\nComparacion completada.")
