import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. Transformadores Personalizados
# ─────────────────────────────────────────────

class DropColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas por nombre. errors='ignore' evita fallos si no existen."""
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class Imputacion(BaseEstimator, TransformerMixin):
    """
    Imputa nulos con la mediana aprendida en fit.
    Se usa mediana (no media) porque las variables financieras son sesgadas.
    """
    COLS = ["saldo_mora", "saldo_total", "saldo_mora_codeudor", "puntaje_datacredito"]

    def fit(self, X, y=None):
        self.medians_ = {
            col: X[col].median()
            for col in self.COLS
            if col in X.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col, median in self.medians_.items():
            X[col] = X[col].fillna(median)
        return X


class NuevasVariables(BaseEstimator, TransformerMixin):
    """
    Features derivadas del EDA validadas por importancia (RandomForest):
    - tiene_mora           : binaria (saldo_mora > 0). Más informativa que el saldo raw.
    - ratio_capital_salario: exposición relativa al ingreso. Aporta info que
                             capital_prestado y salario_cliente solos no capturan.

    Descartadas por redundancia con sus versiones continuas:
    - grupo_edad          (suma 0.034 vs edad_cliente 0.080)
    - tamaño_credito      (suma 0.026 vs capital_prestado 0.094)
    - huella_cat          (suma 0.030 vs huella_consulta 0.061)
    - tiene_mora_codeudor (importancia 0.000)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Binaria de mora
        if "saldo_mora" in X.columns:
            X["tiene_mora"] = (X["saldo_mora"] > 0).astype(int)

        # Ratio capital / salario (evitar división por cero)
        if "capital_prestado" in X.columns and "salario_cliente" in X.columns:
            salario_safe = X["salario_cliente"].replace(0, np.nan)
            X["ratio_capital_salario"] = (X["capital_prestado"] / salario_safe).fillna(0)

        return X


class ToDF(BaseEstimator, TransformerMixin):
    """
    Escala variables numéricas y aplica OHE a categóricas.
    Retorna un DataFrame con nombres de columna para facilitar interpretabilidad.
    """
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_features),
            ("cat", ohe, self.categorical_features),
        ])
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.ct_.transform(X)
        try:
            feat_names = self.ct_.get_feature_names_out()
        except AttributeError:
            feat_names = []
            for name, trans, cols in self.ct_.transformers_:
                if name == "remainder" and trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    feat_names.extend(trans.get_feature_names_out(cols))
                else:
                    feat_names.extend(cols)
        return pd.DataFrame(Xt, columns=feat_names, index=X.index)


# ─────────────────────────────────────────────
# 2. Pipeline Base
# Se aplica sobre el df COMPLETO antes del split.
# Incluye: eliminación de columnas, imputación y creación de features.
# NO incluye escalado ni encoding (eso va en pipeline_ml).
# ─────────────────────────────────────────────

COLS_ALTA_NULIDAD = ["promedio_ingresos_datacredito", "tendencia_ingresos"]

COLS_IRRELEVANTES = [
    "fecha_prestamo",        # sin razón de negocio clara
    "puntaje",               # correlación 0.923 con target → data leakage
    "cuota_pactada",         # correlación 0.76 con capital_prestado
    "saldo_principal",       # correlación 0.73 con saldo_total (más nulos)
    "cant_creditosvigentes", # correlación 0.79 con creditos_sectorFinanciero
    "saldo_mora",            # reemplazada por tiene_mora (binaria más informativa)
    "saldo_mora_codeudor",   # importancia 0.000 → no aporta al modelo
]

pipeline_basemodel = Pipeline(steps=[
    ("drop_nulidad",      DropColumns(cols_to_drop=COLS_ALTA_NULIDAD)),
    ("imputacion",        Imputacion()),
    ("nuevas_variables",  NuevasVariables()),
    ("drop_irrelevantes", DropColumns(cols_to_drop=COLS_IRRELEVANTES)),
])


# ─────────────────────────────────────────────
# 3. Pipeline ML
# Solo escala + encoding. Se aplica DESPUÉS del split
# para evitar data leakage en validación cruzada.
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "puntaje_datacredito", "huella_consulta",
    "saldo_total", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "tiene_mora", "ratio_capital_salario",
]

CATEGORICAL_FEATURES = [
    "tipo_credito", "tipo_laboral",
]

pipeline_ml = Pipeline(steps=[
    ("preprocessor", ToDF(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )),
])


# ─────────────────────────────────────────────
# 4. build_feature_pipeline
# ─────────────────────────────────────────────

TARGET = "Pago_atiempo"


def build_feature_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    1. Elimina outliers extremos de forma explícita (no dentro de un transformer,
       ya que eliminar filas en transform() rompe índices en CV/producción).
    2. Aplica pipeline_basemodel al DataFrame completo.
    3. Hace split estratificado por la variable objetivo (desbalance 95/5%).
    4. Retorna X_train, X_test, y_train, y_test y pipeline_ml listo para usar.

    Nota: pipeline_ml (ToDF) debe fittearse con X_train en model_training.py,
    nunca aquí, para evitar data leakage.
    """
    # Paso 1 – Eliminar outliers extremos antes del pipeline
    n_before = len(df)
    df = df[df["edad_cliente"].between(18, 100) | df["edad_cliente"].isna()].copy()
    df = df[df["puntaje_datacredito"].between(150, 950) | df["puntaje_datacredito"].isna()].copy()
    n_after = len(df)
    print(f"Outliers eliminados: {n_before - n_after} registros")

    # Paso 2 – Pipeline base
    df_clean = pipeline_basemodel.fit_transform(df)

    # Paso 3 – Split estratificado
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Resumen
    print(f"\nDataset procesado : {df_clean.shape[0]} registros | {df_clean.shape[1]} columnas")
    print(f"X_train           : {X_train.shape}")
    print(f"X_test            : {X_test.shape}")
    print(f"Balance target (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")

    return X_train, X_test, y_train, y_test, pipeline_ml


# ─────────────────────────────────────────────
# 5. Ejecución directa
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from gcp_utils import load_data_from_gcp

    print("=" * 55)
    print("   FEATURE ENGINEERING PIPELINE – ft_engineering.py")
    print("=" * 55)

    print("\nCargando dataset desde GCP...")
    df = load_data_from_gcp()
    print(f"   Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        print(f"   Nulos por columna:\n{nulos.to_string()}")

    print("\nAplicando pipeline...")
    X_train, X_test, y_train, y_test, pipe_ml = build_feature_pipeline(df)

    print("\nConjuntos generados:")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   y_train : {y_train.shape}")
    print(f"   y_test  : {y_test.shape}")

    print("\nBalance del target (train):")
    for k, v in y_train.value_counts().items():
        label = "Paga a tiempo" if k == 1 else "Mora"
        print(f"   {label} ({k}): {v} registros ({v / len(y_train) * 100:.1f}%)")

    print("\nColumnas finales (X_train):")
    for col in X_train.columns:
        print(f"   - {col}")