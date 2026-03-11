import io
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import classification_report

from ft_engineering import build_feature_pipeline


# ─────────────────────────────────────────────
# _cargar_referencia()
# Carga y cachea X_train como dataset de referencia.
# Evita recargar y reprocesar el Excel en cada llamada
# a detectar_data_drift() o monitoreo_periodico().
# ─────────────────────────────────────────────

_referencia_cache = None

def _cargar_referencia():
    """
    Carga X_train una sola vez y lo cachea en memoria.
    En producción real esto vendría del DWH/Datalake.
    """
    global _referencia_cache
    if _referencia_cache is None:
        df_ref = pd.read_excel("Base_de_datos.xlsx")
        X_train, _, _, _, _ = build_feature_pipeline(df_ref)
        _referencia_cache = X_train
        print("Dataset de referencia cargado y cacheado.")
    return _referencia_cache


# ─────────────────────────────────────────────
# detectar_data_drift()
# Compara distribuciones entre datos de referencia
# y datos de producción usando el test KS.
# ─────────────────────────────────────────────

def detectar_data_drift(df_produccion: pd.DataFrame,
                        umbral_pvalue: float = 0.05) -> pd.DataFrame:
    """
    Detecta Data Drift comparando la distribución de cada variable numérica
    entre el dataset de referencia (train) y los datos de producción.

    Usa el test de Kolmogorov-Smirnov (KS):
    - p-value < umbral → distribuciones significativamente distintas → drift
    - p-value >= umbral → no hay evidencia de drift

    Args:
        df_produccion  : DataFrame con datos nuevos del endpoint
        umbral_pvalue  : nivel de significancia (default 0.05)

    Returns:
        DataFrame con estadístico KS, p-value y flag de drift por variable
    """
    X_train = _cargar_referencia()

    # Solo variables numéricas comunes entre referencia y producción
    cols_numericas = X_train.select_dtypes(include=np.number).columns.tolist()
    cols_comunes   = [c for c in cols_numericas if c in df_produccion.columns]

    if not cols_comunes:
        return pd.DataFrame()

    # Convertir a numérico — evita error entre int y str cuando
    # tipo_credito llega como string desde el endpoint
    df_produccion = df_produccion.copy()
    for col in cols_comunes:
        df_produccion[col] = pd.to_numeric(df_produccion[col], errors="coerce")

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
            "drift_detectado" : bool(pvalue < umbral_pvalue),
            "media_referencia": round(ref_vals.mean(),  4),
            "media_produccion": round(prod_vals.mean(), 4),
        })

    df_reporte = pd.DataFrame(resultados)
    if df_reporte.empty:
        return df_reporte

    df_reporte = df_reporte.astype({
        "ks_statistic"    : float,
        "p_value"         : float,
        "drift_detectado" : bool,
        "media_referencia": float,
        "media_produccion": float,
    })

    return df_reporte


# ─────────────────────────────────────────────
# monitoreo_periodico()
# Simula un ciclo de monitoreo con una muestra
# de los datos de referencia como proxy de producción.
# En producción real se conectaría al DWH/Datalake.
# ─────────────────────────────────────────────

def monitoreo_periodico(fraccion_muestra: float = 0.1,
                        umbral_pvalue: float = 0.05) -> io.BytesIO:
    """
    Simula el monitoreo periódico del modelo:
    1. Toma una muestra aleatoria del dataset como proxy de producción
    2. Pasa la muestra por el modelo y obtiene predicciones
    3. Detecta drift entre referencia y muestra usando test KS
    4. Genera un reporte visual de 3 gráficos

    Args:
        fraccion_muestra : fracción del X_test a usar como muestra (default 10%)
        umbral_pvalue    : nivel de significancia para KS test (default 0.05)

    Returns:
        io.BytesIO: buffer PNG con el reporte visual
    """
    # Cargar modelo y datos
    model  = joblib.load("mejor_modelo.pkl")
    df     = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    # Muestra de producción simulada desde X_test
    n_muestra  = max(50, int(len(X_test) * fraccion_muestra))
    df_muestra = X_test.sample(n=n_muestra, random_state=42)
    y_muestra  = y_test.loc[df_muestra.index]

    # Predicciones sobre la muestra
    y_pred  = model.predict(df_muestra)
    y_proba = model.predict_proba(df_muestra)[:, 0]  # prob mora

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

    variables_con_drift = reporte_drift[
        reporte_drift["drift_detectado"]
    ]["variable"].tolist()
    print(f"\nVariables con drift: {variables_con_drift if variables_con_drift else 'Ninguna'}")

    # ── Gráfico de monitoreo ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Distribución de predicciones
    conteo = pd.Series(y_pred).map({1: "Paga a tiempo", 0: "Mora"}).value_counts()
    axes[0].bar(conteo.index, conteo.values,
                color=["steelblue", "coral"], edgecolor="white")
    for i, (label, val) in enumerate(conteo.items()):
        axes[0].text(i, val + 0.5, str(val), ha="center", fontsize=10)
    axes[0].set_title("Distribución de predicciones\n(muestra producción)", fontweight="bold")
    axes[0].set_ylabel("Cantidad")
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. KS statistic por variable (top 10)
    # Colores vectorizados — evita loop con .loc por cada variable
    drift_plot = reporte_drift.set_index("variable")["ks_statistic"].sort_values(
        ascending=False
    ).head(10)
    drift_flags = reporte_drift.set_index("variable")["drift_detectado"]
    colores     = ["coral" if drift_flags.get(v, False) else "steelblue"
                   for v in drift_plot.index]

    axes[1].barh(drift_plot.index, drift_plot.values, color=colores, edgecolor="white")
    axes[1].axvline(x=0.1, color="red", linestyle="--", alpha=0.5, label="Referencia 0.1")
    axes[1].set_title("KS Statistic por variable\n(rojo = drift detectado)", fontweight="bold")
    axes[1].set_xlabel("KS Statistic")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. Comparación de medias: referencia vs producción
    diff = (
        reporte_drift.set_index("variable")["media_produccion"] -
        reporte_drift.set_index("variable")["media_referencia"]
    ).abs().sort_values(ascending=False).head(8)

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
        with open("model_monitoring.png", "wb") as f:
            f.write(buf.getvalue())
        print("\nReporte guardado: model_monitoring.png")