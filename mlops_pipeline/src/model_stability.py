import os
import sys
import warnings
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    roc_auc_score, recall_score, precision_score,
)

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
SRC_PATH   = _HERE
MODEL_PATH = os.path.join(_HERE, "mejor_modelo.pkl")
TARGET     = "Pago_atiempo"
N_FOLDS    = 5
STD_THRESHOLD = 0.05

METRICS = ["accuracy", "f1", "roc_auc", "recall", "precision"]


# ─────────────────────────────────────────────
# 1 & 2. Datos desde GCP + pipeline_basemodel
# ─────────────────────────────────────────────
def load_full_dataset():
    sys.path.insert(0, SRC_PATH)
    from gcp_utils import load_data_from_gcp
    from ft_engineering import pipeline_basemodel

    print("Descargando datos desde GCP...")
    df = load_data_from_gcp()

    print("Aplicando limpieza y pipeline base...")
    df = df[df["edad_cliente"].between(18, 100) | df["edad_cliente"].isna()].copy()
    df = df[df["puntaje_datacredito"].between(150, 950) | df["puntaje_datacredito"].isna()].copy()

    df_clean = pipeline_basemodel.fit_transform(df)
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET].astype(int)
    print(f"Dataset listo: {X.shape[0]} registros | {X.shape[1]} features")
    return X, y


# ─────────────────────────────────────────────
# 3. Validación cruzada 5 folds
# ─────────────────────────────────────────────
def run_cross_validation(X, y):
    print(f"\nEjecutando validacion cruzada con {N_FOLDS} folds...")
    model = joblib.load(MODEL_PATH)

    def roc_auc_scorer(estimator, X, y):
        try:
            proba = estimator.predict_proba(X)[:, 1]
            return roc_auc_score(y, proba)
        except Exception:
            pred = estimator.predict(X)
            return roc_auc_score(y, pred)

    scorers = {
        "accuracy":  make_scorer(accuracy_score),
        "f1":        make_scorer(f1_score,        zero_division=0),
        "roc_auc":   roc_auc_scorer,
        "recall":    make_scorer(recall_score,     zero_division=0),
        "precision": make_scorer(precision_score,  zero_division=0),
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers, return_train_score=False)

    # Organizar resultados por métrica
    results = {m: cv_results[f"test_{m}"] for m in METRICS}
    return results


# ─────────────────────────────────────────────
# 4. Boxplot
# ─────────────────────────────────────────────
def plot_stability_boxplot(results: dict, output_path: str):
    data   = [results[m] for m in METRICS]
    labels = [m.replace("_", "\n") for m in METRICS]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Umbral 0.5")
    ax.set_title(f"Estabilidad del Modelo Local — Validacion Cruzada ({N_FOLDS} folds)")
    ax.set_ylabel("Valor de la metrica")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Boxplot guardado: {output_path}")


# ─────────────────────────────────────────────
# 5. Tabla resumen
# ─────────────────────────────────────────────
def print_summary_table(results: dict):
    print("\n" + "=" * 65)
    print(f"  {'METRICA':<12} {'FOLD 1':>7} {'FOLD 2':>7} {'FOLD 3':>7} {'FOLD 4':>7} {'FOLD 5':>7}   MEDIA ± STD")
    print("=" * 65)
    for m in METRICS:
        vals  = results[m]
        media = np.mean(vals)
        std   = np.std(vals)
        folds = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {m:<12} {folds}   {media:.4f} ± {std:.4f}")
    print("=" * 65)


# ─────────────────────────────────────────────
# 6. Conclusión de estabilidad
# ─────────────────────────────────────────────
def print_stability_conclusion(results: dict):
    print("\n--- Conclusion de estabilidad (umbral std < 0.05) ---")
    all_stable = True
    for m in METRICS:
        std    = np.std(results[m])
        stable = std < STD_THRESHOLD
        estado = "ESTABLE" if stable else "INESTABLE"
        if not stable:
            all_stable = False
        print(f"  {m:<12}: std={std:.4f}  ->  {estado}")

    print("\n" + ("El modelo es ESTABLE en todos los folds." if all_stable
                  else "El modelo presenta INESTABILIDAD en algunas metricas."))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   MODEL STABILITY — model_stability.py")
    print("=" * 60)

    X, y = load_full_dataset()
    results = run_cross_validation(X, y)
    print_summary_table(results)
    plot_stability_boxplot(results, "stability_boxplot.png")
    print_stability_conclusion(results)
