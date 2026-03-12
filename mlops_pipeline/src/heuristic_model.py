import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, cross_validate, learning_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. Modelo Heurístico de Riesgo Crediticio
# ─────────────────────────────────────────────

class HeuristicModel(BaseEstimator, ClassifierMixin):
    """
    Modelo baseline basado en reglas de negocio para predecir incumplimiento de crédito.

    Reglas (aplicadas en orden de prioridad mediante vectorización):
    1. Cliente con mora activa → alto riesgo (predice 0 = no paga)
    2. Puntaje Datacrédito muy bajo (< 600) → alto riesgo
    3. Cliente joven (< 30) + ratio capital/salario alto (> 3) → riesgo elevado
    4. Huella de consulta alta (> 8) + puntaje bajo (< 750) → riesgo moderado
    5. Independiente + puntaje < 700 → riesgo moderado
    6. Caso por defecto → paga a tiempo (1)

    Nota: se usa vectorización con pandas en lugar de iterrows() para
    mayor eficiencia en producción con datasets de 10.000+ registros.
    """

    def __init__(
        self,
        puntaje_muy_bajo: float = 600,
        puntaje_moderado: float = 750,
        puntaje_independiente: float = 700,
        edad_joven: int = 30,
        ratio_alto: float = 3.0,
        huella_alta: int = 8,
    ):
        self.puntaje_muy_bajo = puntaje_muy_bajo
        self.puntaje_moderado = puntaje_moderado
        self.puntaje_independiente = puntaje_independiente
        self.edad_joven = edad_joven
        self.ratio_alto = ratio_alto
        self.huella_alta = huella_alta

    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """
        Predicción vectorizada — evita iterrows() para mayor eficiencia.
        Las reglas se aplican de menor a mayor prioridad (la última en
        escribir gana), por lo que la Regla 1 se aplica al final para
        asegurar que tenga la mayor precedencia.
        """
        # Default: todos pagan a tiempo
        pred = pd.Series(1, index=X.index)

        # Regla 5: Independiente con puntaje bajo → mora
        if "tipo_laboral" in X.columns and "puntaje_datacredito" in X.columns:
            mask = (
                (X["tipo_laboral"].astype(str) == "Independiente") &
                (X["puntaje_datacredito"] < self.puntaje_independiente)
            )
            pred[mask] = 0

        # Regla 4: Huella alta + puntaje moderado → mora
        if "huella_consulta" in X.columns and "puntaje_datacredito" in X.columns:
            mask = (
                (X["huella_consulta"] > self.huella_alta) &
                (X["puntaje_datacredito"] < self.puntaje_moderado)
            )
            pred[mask] = 0

        # Regla 3: Joven + ratio capital/salario alto → mora
        if "edad_cliente" in X.columns and "ratio_capital_salario" in X.columns:
            mask = (
                (X["edad_cliente"] < self.edad_joven) &
                (X["ratio_capital_salario"] > self.ratio_alto)
            )
            pred[mask] = 0

        # Regla 2: Puntaje Datacrédito muy bajo → mora
        if "puntaje_datacredito" in X.columns:
            pred[X["puntaje_datacredito"] < self.puntaje_muy_bajo] = 0

        # Regla 1 (máxima prioridad): mora activa → mora
        if "tiene_mora" in X.columns:
            pred[X["tiene_mora"] == 1] = 0

        return pred.values


# ─────────────────────────────────────────────
# 2. Evaluación del modelo heurístico
# ─────────────────────────────────────────────

def evaluate_heuristic(model, X_train, X_test, y_train, y_test, n_splits: int = 10):
    """
    Evalúa el modelo heurístico con validación cruzada y curva de aprendizaje.

    Evalúa tres dimensiones:
    - Performance  : métricas en train y test
    - Consistency  : KFold cross-validation
    - Scalability  : curva de aprendizaje (recall vs tamaño de muestra)

    Retorna
    -------
    cv_results_df : pd.DataFrame con scores por fold.
    metrics_df    : pd.DataFrame con resumen train vs CV.
    """
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

    scoring_metrics = {
        "accuracy" : "accuracy",
        "f1"       : make_scorer(f1_score,        pos_label=0, zero_division=0),
        "precision": make_scorer(precision_score,  pos_label=0, zero_division=0),
        "recall"   : make_scorer(recall_score,     pos_label=0, zero_division=0),
    }
    kfold      = KFold(n_splits=n_splits)
    model_pipe = Pipeline(steps=[("model", model)])

    cv_output = cross_validate(
        model_pipe, X_train, y_train,
        cv=kfold,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1,
    )

    cv_results    = {m: cv_output[f"test_{m}"]              for m in scoring_metrics}
    train_results = {m: np.mean(cv_output[f"train_{m}"]) for m in scoring_metrics}
    cv_results_df = pd.DataFrame(cv_results)

    # ── Variabilidad entre métricas ────────────
    cv_results_df.plot.box(
        title="Cross Validation Boxplot – Modelo Heurístico",
        ylabel="Score"
    )
    plt.tight_layout()
    plt.show()

    # ── Train vs CV ───────────────────────────
    metric_names = list(scoring_metrics.keys())
    metrics_df = pd.DataFrame({
        "Metric"      : metric_names,
        "Train Score" : [train_results[m] for m in metric_names],
        "CV Mean"     : [cv_results_df[m].mean() for m in metric_names],
        "CV Std"      : [cv_results_df[m].std()  for m in metric_names],
    })

    metrics_df.plot(
        kind="bar", x="Metric", y=["Train Score", "CV Mean"],
        yerr="CV Std",
        title="Training vs Cross-Validation – Modelo Heurístico",
        ylabel="Score", capsize=4
    )
    plt.tight_layout()
    plt.show()

    # ── Matriz de confusión ───────────────────
    model_pipe.fit(X_train, y_train)
    y_pred = model_pipe.predict(X_test)
    print("\nReporte de clasificación – Modelo Heurístico:")
    print(classification_report(y_test, y_pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Matriz de Confusión – Modelo Heurístico")
    plt.show()

    # ── Curva de aprendizaje ──────────────────
    common_params = {
        "X"           : X_train,
        "y"           : y_train,
        "train_sizes" : np.linspace(0.1, 1.0, 5),
        "cv"          : ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        "n_jobs"      : -1,
        "return_times": False,
    }
    train_sizes, train_scores, test_scores = learning_curve(
        model_pipe, scoring="recall", **common_params
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
    ax.plot(train_sizes, test_scores.mean(axis=1), "o-", color="orange", label="CV score")
    ax.fill_between(
        train_sizes,
        train_scores.mean(1) - train_scores.std(1),
        train_scores.mean(1) + train_scores.std(1),
        alpha=0.3
    )
    ax.fill_between(
        train_sizes,
        test_scores.mean(1) - test_scores.std(1),
        test_scores.mean(1) + test_scores.std(1),
        alpha=0.3, color="orange"
    )
    ax.set_title(f"Curva de Aprendizaje – {model.__class__.__name__}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Recall")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return cv_results_df, metrics_df


# ─────────────────────────────────────────────
# 3. Ejecución directa
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from ft_engineering import build_feature_pipeline

    print("=" * 55)
    print("   MODELO HEURÍSTICO – heuristic_model.py")
    print("=" * 55)

    print("\nCargando y procesando datos...")
    df = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    print("\nEntrenando modelo heurístico...")
    model = HeuristicModel()
    print(f"   Reglas configuradas:")
    print(f"   - Puntaje muy bajo    : < {model.puntaje_muy_bajo}")
    print(f"   - Puntaje moderado    : < {model.puntaje_moderado}")
    print(f"   - Puntaje independ.   : < {model.puntaje_independiente}")
    print(f"   - Edad joven          : < {model.edad_joven} años")
    print(f"   - Ratio capital alto  : > {model.ratio_alto}")
    print(f"   - Huella alta         : > {model.huella_alta} consultas")

    print("\nEvaluando modelo...")
    cv_df, metrics_df = evaluate_heuristic(model, X_train, X_test, y_train, y_test)

    print("\nResumen de métricas:")
    print("-" * 55)
    for _, row in metrics_df.iterrows():
        print(
            f"   {row['Metric']:<12} | "
            f"Train: {row['Train Score']:.3f} | "
            f"CV Mean: {row['CV Mean']:.3f} ± {row['CV Std']:.3f}"
        )
    print("=" * 55)