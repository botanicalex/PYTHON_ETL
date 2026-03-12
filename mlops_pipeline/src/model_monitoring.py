import io
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import classification_report

from typing import Optional

from functools import lru_cache

from ft_engineering import build_feature_pipeline

os.makedirs("outputs", exist_ok=True)


@lru_cache(maxsize=1)
def _get_reference_X_train() -> pd.DataFrame:
    """
    Carga y procesa el dataset de referencia solo una vez por proceso.
    El resultado se cachea para evitar I/O repetido en cada llamada a detectar_data_drift.
    """
    df_ref = pd.read_excel("Base_de_datos.xlsx")
    X_train, *_ = build_feature_pipeline(df_ref)
    return X_train


# ─────────────────────────────────────────────
# detectar_data_drift
# Compara distribuciones entre datos de referencia
# y datos de producción usando el test KS.
# ─────────────────────────────────────────────

def detectar_data_drift(df_produccion: pd.DataFrame,
                        umbral_pvalue: float = 0.05) -> pd.DataFrame:
    """
    Detecta Data Drift comparando la distribución de cada variable numérica
    entre el dataset de referencia (train) y los datos de producción.

    Usa el test de Kolmogorov-Smirnov (KS):
    - p-value < umbral → las distribuciones son significativamente distintas → drift
    - p-value >= umbral → no hay evidencia de drift

    Args:
        df_produccion  : DataFrame con datos nuevos del endpoint
        umbral_pvalue  : nivel de significancia (default 0.05)

    Returns:
        DataFrame con estadístico KS, p-value y flag de drift por variable
    """
    # Cargar datos de referencia (cacheados en memoria)
    X_train = _get_reference_X_train()

    # Solo variables numéricas comunes
    cols_numericas = X_train.select_dtypes(include=np.number).columns.tolist()
    cols_comunes   = [c for c in cols_numericas if c in df_produccion.columns]

    if not cols_comunes:
        return pd.DataFrame()

    if len(df_produccion) < 2:
        return pd.DataFrame()  # muestra insuficiente para KS test (mínimo 2 registros)

    resultados = []
    for col in cols_comunes:
        ref_vals  = X_train[col].dropna()
        prod_vals = df_produccion[col].dropna()

        if len(prod_vals) < 2:
            continue

        stat, pvalue = ks_2samp(ref_vals, prod_vals)
        resultados.append({
            "variable"        : col,
            "ks_statistic"    : round(stat,   4),
            "p_value"         : round(pvalue, 4),
            "drift_detectado" : pvalue < umbral_pvalue,
            "media_referencia": round(ref_vals.mean(),  4),
            "media_produccion": round(prod_vals.mean(), 4),
        })

    df_reporte = pd.DataFrame(resultados)
    df_reporte["ks_statistic"]     = df_reporte["ks_statistic"].astype(float)
    df_reporte["p_value"]          = df_reporte["p_value"].astype(float)
    df_reporte["drift_detectado"]  = df_reporte["drift_detectado"].astype(bool)
    df_reporte["media_referencia"] = df_reporte["media_referencia"].astype(float)
    df_reporte["media_produccion"] = df_reporte["media_produccion"].astype(float)
    return df_reporte


# ─────────────────────────────────────────────
# monitoreo_periodico
# Simula un ciclo de monitoreo con una muestra
# de los datos de referencia como proxy de producción.
# En producción real se conectaría al DWH/Datalake.
# ─────────────────────────────────────────────

def monitoreo_periodico(fraccion_muestra: float = 0.1,
                        umbral_pvalue: float = 0.05) -> Optional[io.BytesIO]:
    """
    Simula el monitoreo periódico del modelo:
    1. Toma una muestra aleatoria del dataset como proxy de datos de producción
    2. Pasa la muestra por el modelo y obtiene predicciones
    3. Detecta drift entre referencia y muestra
    4. Genera un reporte visual

    Args:
        fraccion_muestra : fracción del dataset a usar como muestra (default 10%)
        umbral_pvalue    : nivel de significancia para KS test (default 0.05)

    Returns:
        io.BytesIO: buffer PNG con el reporte visual
    """
    # Cargar modelo y datos
    try:
        model = joblib.load("mejor_modelo.pkl")
    except FileNotFoundError:
        print("Error: mejor_modelo.pkl no encontrado.")
        return None
    df    = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    # Muestra de producción (simulada)
    n_muestra    = max(50, int(len(X_test) * fraccion_muestra))
    df_muestra   = X_test.sample(n=n_muestra, random_state=42)

    # Predicciones sobre la muestra
    y_pred       = model.predict(df_muestra)
    y_proba      = model.predict_proba(df_muestra)[:, 0]

    # Tabla de predicciones
    df_resultado = df_muestra.copy()
    df_resultado["prediccion"]        = y_pred
    df_resultado["probabilidad_mora"] = y_proba.round(4)
    df_resultado["etiqueta"]          = df_resultado["prediccion"].map(
        {1: "Paga a tiempo", 0: "Mora"}
    )

    print(f"\nMuestra de producción: {n_muestra} registros")
    print(f"Predicciones:\n{df_resultado['etiqueta'].value_counts().to_string()}")

    # Detección de drift
    reporte_drift = detectar_data_drift(df_muestra, umbral_pvalue=umbral_pvalue)

    print("\n=== REPORTE DE DATA DRIFT ===")
    print(reporte_drift.to_string())

    variables_con_drift = reporte_drift[reporte_drift["drift_detectado"]]["variable"].tolist()
    print(f"\nVariables con drift detectado: {variables_con_drift if variables_con_drift else 'Ninguna'}")

    # ── Gráfico de monitoreo ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Distribución de predicciones
    conteo = pd.Series(y_pred).map({1: "Paga a tiempo", 0: "Mora"}).value_counts()
    color_map = {"Paga a tiempo": "steelblue", "Mora": "coral"}
    colores_barras = [color_map.get(label, "gray") for label in conteo.index]
    axes[0].bar(conteo.index, conteo.values, color=colores_barras, edgecolor="white")
    for i, (label, val) in enumerate(conteo.items()):
        axes[0].text(i, val + 0.5, str(val), ha="center", fontsize=10)
    axes[0].set_title("Distribución de predicciones\n(muestra producción)", fontweight="bold")
    axes[0].set_ylabel("Cantidad")
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. KS statistic por variable (top 10)
    drift_plot = reporte_drift.set_index("variable")["ks_statistic"].sort_values(ascending=False).head(10)
    colores    = ["coral" if reporte_drift.set_index("variable").loc[v, "drift_detectado"] else "steelblue"
                  for v in drift_plot.index]
    axes[1].barh(drift_plot.index, drift_plot.values, color=colores, edgecolor="white")
    axes[1].axvline(x=0.1, color="red", linestyle="--", alpha=0.5, label="Referencia 0.1")
    axes[1].set_title("KS Statistic por variable\n(rojo = drift detectado)", fontweight="bold")
    axes[1].set_xlabel("KS Statistic")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. Comparación de medias: referencia vs producción
    reporte_idx = reporte_drift.set_index("variable")
    diff = (reporte_idx["media_produccion"] - reporte_idx["media_referencia"]).abs()
    diff = diff.sort_values(ascending=False).head(8)
    axes[2].barh(diff.index, diff.values, color="mediumpurple", edgecolor="white")
    axes[2].set_title("Diferencia de medias\n(referencia vs producción)", fontweight="bold")
    axes[2].set_xlabel("Diferencia absoluta")
    axes[2].spines[["top", "right"]].set_visible(False)

    hay_drift = len(variables_con_drift) > 0
    estado    = "[!] DRIFT DETECTADO" if hay_drift else "[OK] Sin drift significativo"
    plt.suptitle(
        f"Monitoreo periódico — {estado}  |  Muestra: {n_muestra} registros",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

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
    print("   MODEL MONITORING – model_monitoring.py")
    print("=" * 55)

    buf = monitoreo_periodico(fraccion_muestra=0.1)
    if buf:
        with open("outputs/model_monitoring.png", "wb") as f:
            f.write(buf.getvalue())
        print("\nReporte guardado: outputs/model_monitoring.png")