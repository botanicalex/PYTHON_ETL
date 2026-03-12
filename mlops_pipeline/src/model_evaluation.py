import io
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
)

from typing import Optional

from ft_engineering import build_feature_pipeline

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# evaluation()
# Genera un dashboard de métricas del modelo desplegado.
# Retorna un buffer PNG para ser servido por el endpoint /evaluation.
# ─────────────────────────────────────────────

def evaluation() -> Optional[io.BytesIO]:
    """
    Carga el modelo desplegado y los datos de test,
    genera un dashboard con:
    - Tabla de métricas (precision, recall, f1, roc-auc)
    - Matriz de confusión
    - Curva ROC
    - Curva Precision-Recall

    Returns:
        io.BytesIO: buffer con la imagen PNG lista para servir
    """
    # Cargar modelo
    try:
        model = joblib.load("mejor_modelo.pkl")
    except FileNotFoundError:
        print("Error: mejor_modelo.pkl no encontrado.")
        return None

    # Cargar datos
    df = pd.read_excel("Base_de_datos.xlsx")
    _, X_test, _, y_test, _ = build_feature_pipeline(df)

    # Predicciones
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 0]  # probabilidad clase mora (0)

    # ── Métricas ──────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=["Mora", "Paga a tiempo"],
        output_dict=True,
        zero_division=0,
    )

    metricas_mora = {
        "Precision (mora)" : round(report["Mora"]["precision"], 4),
        "Recall (mora)"    : round(report["Mora"]["recall"],    4),
        "F1 (mora)"        : round(report["Mora"]["f1-score"],  4),
        "ROC-AUC"          : round(roc_auc_score(y_test, y_proba, pos_label=0), 4),
        "Accuracy"         : round(report["accuracy"], 4),
        "Soporte mora"     : int(report["Mora"]["support"]),
    }

    print("=== MÉTRICAS DEL MODELO DESPLEGADO ===")
    for k, v in metricas_mora.items():
        print(f"   {k:25s}: {v}")

    # ── Dashboard ─────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Tabla de métricas
    ax_tabla = fig.add_subplot(gs[0, 0])
    ax_tabla.axis("off")
    tabla_data = [[k, str(v)] for k, v in metricas_mora.items()]
    tabla = ax_tabla.table(
        cellText=tabla_data,
        colLabels=["Métrica", "Valor"],
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.8)
    ax_tabla.set_title("Métricas del modelo desplegado", fontweight="bold", pad=12)

    # 2. Matriz de confusión
    ax_cm = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
        xticklabels=["Mora", "Paga"], yticklabels=["Mora", "Paga"]
    )
    ax_cm.set_title("Matriz de confusión", fontweight="bold")
    ax_cm.set_ylabel("Real")
    ax_cm.set_xlabel("Predicho")

    # 3. Barras de métricas clave
    ax_bar = fig.add_subplot(gs[0, 2])
    metricas_plot = {
        "Precision": metricas_mora["Precision (mora)"],
        "Recall"   : metricas_mora["Recall (mora)"],
        "F1"       : metricas_mora["F1 (mora)"],
        "ROC-AUC"  : metricas_mora["ROC-AUC"],
    }
    bars = ax_bar.bar(
        metricas_plot.keys(), metricas_plot.values(),
        color=["steelblue", "coral", "seagreen", "mediumpurple"],
        edgecolor="white",
    )
    for bar, val in zip(bars, metricas_plot.values()):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9
        )
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("Métricas clase mora", fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)

    # 4. Curva ROC
    ax_roc = fig.add_subplot(gs[1, 0:2])
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=0)
    ax_roc.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"ROC curve (AUC = {metricas_mora['ROC-AUC']:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random classifier")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Curva ROC", fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.spines[["top", "right"]].set_visible(False)

    # 5. Curva Precision-Recall
    ax_pr = fig.add_subplot(gs[1, 2])
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba, pos_label=0)
    ax_pr.plot(recall_curve, precision_curve, color="coral", lw=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curva Precision-Recall\n(clase mora)", fontweight="bold")
    ax_pr.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"Evaluación del modelo desplegado — {model.steps[-1][1].__class__.__name__}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    # Retornar buffer para el endpoint /evaluation
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plt.close()

    return buffer


# ─────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("   MODEL EVALUATION – model_evaluation.py")
    print("=" * 55)

    buf = evaluation()
    if buf:
        with open("outputs/model_evaluation.png", "wb") as f:
            f.write(buf.getvalue())
        print("\nGráfico guardado: outputs/model_evaluation.png")