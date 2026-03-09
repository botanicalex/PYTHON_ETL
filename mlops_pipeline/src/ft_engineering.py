import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# 1. Transformadores Personalizados

class ColumnasNulos(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class LimpiarTendencia(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "tendencia_ingresos" in X.columns:
            def _limpiar(val):
                if val in ["Creciente", "Estable", "Decreciente"]:
                    return val
                return np.nan
            X["tendencia_ingresos"] = X["tendencia_ingresos"].apply(_limpiar)
        return X


class Imputacion(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_saldo_mora_ = X["saldo_mora"].median() if "saldo_mora" in X.columns else 0
        self.median_saldo_total_ = X["saldo_total"].median() if "saldo_total" in X.columns else 0
        self.median_saldo_mora_cod_ = X["saldo_mora_codeudor"].median() if "saldo_mora_codeudor" in X.columns else 0
        self.mean_puntaje_ = X["puntaje_datacredito"].mean() if "puntaje_datacredito" in X.columns else 0
        return self

    def transform(self, X):
        X = X.copy()
        if "saldo_mora" in X.columns:
            X["saldo_mora"] = X["saldo_mora"].fillna(self.median_saldo_mora_)
        if "saldo_total" in X.columns:
            X["saldo_total"] = X["saldo_total"].fillna(self.median_saldo_total_)
        if "saldo_mora_codeudor" in X.columns:
            X["saldo_mora_codeudor"] = X["saldo_mora_codeudor"].fillna(self.median_saldo_mora_cod_)
        if "puntaje_datacredito" in X.columns:
            X["puntaje_datacredito"] = X["puntaje_datacredito"].fillna(self.mean_puntaje_)
        return X


class Outliers(BaseEstimator, TransformerMixin):
    """
    Elimina filas con outliers extremos.
    Solo se usa en pipeline_basemodel, que se aplica sobre el df completo
    ANTES del split — nunca dentro de CV.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "edad_cliente" in X.columns:
            X = X[(X["edad_cliente"] >= 18) & (X["edad_cliente"] <= 100)]
        if "puntaje_datacredito" in X.columns:
            X = X[(X["puntaje_datacredito"] >= 150) & (X["puntaje_datacredito"] <= 950)]
        return X


class NuevasVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "saldo_mora" in X.columns:
            X["tiene_mora"] = (X["saldo_mora"] > 0).astype(int)
        if "saldo_mora_codeudor" in X.columns:
            X["tiene_mora_codeudor"] = (X["saldo_mora_codeudor"] > 0).astype(int)
        if "capital_prestado" in X.columns and "salario_cliente" in X.columns:
            X["ratio_capital_salario"] = X["capital_prestado"] / X["salario_cliente"].replace(0, np.nan)
            X["ratio_capital_salario"] = X["ratio_capital_salario"].fillna(0)
        return X


class ToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X


class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class ToDF(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ct_ = None

    def fit(self, X, y=None):
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", ohe, self.categorical_features),
            ]
        )
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


# 2. Pipeline Base
# Limpieza completa: se aplica sobre el df ANTES del split

pipeline_basemodel = Pipeline(steps=[
    ("eliminar_nulos",        ColumnasNulos(cols_to_drop=["promedio_ingresos_datacredito", "tendencia_ingresos"])),
    ("limpiar_tendencia",     LimpiarTendencia()),
    ("imputacion",            Imputacion()),
    ("outliers",              Outliers()),
    ("nuevas_variables",      NuevasVariables()),
    ("to_category",           ToCategory(cols=["tipo_credito", "tipo_laboral"])),
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop=[
        "fecha_prestamo",
        "puntaje",
        "cuota_pactada",
        "saldo_principal",
        "cant_creditosvigentes",
        "saldo_mora",
        "saldo_mora_codeudor",
    ])),
])


# 3. Pipeline ML
# FIX: ya NO incluye pipeline_basemodel adentro, porque build_feature_pipeline
# ya lo aplicó antes del split. Así evitamos el doble preprocesamiento.
# Solo hace escalado + encoding (ToDF) sobre datos ya limpios.

numeric_features = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "puntaje_datacredito", "huella_consulta",
    "saldo_total", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "tiene_mora", "tiene_mora_codeudor", "ratio_capital_salario",
]

categorical_features = ["tipo_credito", "tipo_laboral"]

pipeline_ml = Pipeline(steps=[
    ("preprocessor", ToDF(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )),
])


# 4. build_feature_pipeline

TARGET = "Pago_atiempo"

def build_feature_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Aplica el pipeline base al DataFrame completo, hace el split estratificado
    y retorna los conjuntos listos para modelamiento.

    pipeline_ml retornado contiene SOLO ToDF (escalado + encoding), ya que
    pipeline_basemodel se aplicó aquí. Esto evita el doble preprocesamiento
    en build_model.

    Retorna
    -------
    X_train, X_test, y_train, y_test, pipeline_ml
    """
    df_clean = pipeline_basemodel.fit_transform(df)

    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"Dataset procesado: {df_clean.shape[0]} registros, {df_clean.shape[1]} columnas")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Balance target train: {y_train.value_counts(normalize=True).round(3).to_dict()}")

    return X_train, X_test, y_train, y_test, pipeline_ml


# 5. Ejecución directa

if __name__ == "__main__":
    print("=" * 55)
    print("   FEATURE ENGINEERING PIPELINE – ft_engineering.py")
    print("=" * 55)

    print("\nCargando dataset...")
    df = pd.read_excel("Base_de_datos.xlsx")
    print(f"   Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f"   Nulos por columna:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    print("\nAplicando pipeline base...")
    X_train, X_test, y_train, y_test, pipe_ml = build_feature_pipeline(df)

    print("\nConjuntos generados:")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   y_train : {y_train.shape}")
    print(f"   y_test  : {y_test.shape}")

    print("\nBalance del target (train):")
    vc = y_train.value_counts()
    for k, v in vc.items():
        label = "Paga a tiempo" if k == 1 else "Mora"
        print(f"   {label} ({k}): {v} registros ({v/len(y_train)*100:.1f}%)")

    print("\nColumnas del dataset procesado (X_train):")
    for col in X_train.columns:
        print(f"   - {col}")