import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    KFold, cross_val_score, learning_curve, ShuffleSplit
)
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# 1. Funciones principales

def summarize_classification(y_true, y_pred, y_proba=None, model_name: str = "Modelo") -> pd.DataFrame:
    """
    Genera un resumen de métricas de clasificación para un modelo dado.

    FIX: Se usan métricas macro-averaged para evaluar correctamente en datos
         desbalanceados — evita que la clase mayoritaria (clase 1) infle los
         resultados.
    FIX: roc_auc ahora usa probabilidades (y_proba) cuando están disponibles,
         en lugar de predicciones binarias, para un AUC real y significativo.
    """
    # FIX: average='macro' trata ambas clases por igual
    summary = {
        "model":          model_name,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":   round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        # Métricas específicas para clase 0 (mora) — la más importante en crédito
        "precision_mora": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_mora":    round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "f1_mora":        round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
    }

    # FIX: AUC con probabilidades si están disponibles
    if y_proba is not None:
        try:
            summary["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            summary["roc_auc"] = "N/A"
    else:
        summary["roc_auc"] = "N/A (sin proba)"

    return pd.DataFrame([summary])


def build_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline_ml,
    n_splits: int = 10,
    model_name: str = None,
    use_smote: bool = True,
) -> dict:
    """
    Entrena un modelo dentro del pipeline ML con SMOTE opcional,
    evalúa con validación cruzada y genera gráficos.

    FIX: pipeline_ml recibido ya es solo ToDF (escalado + encoding).
    FIX: SMOTE se integra al ImbPipeline para aplicarse solo en cada
         fold del CV, nunca sobre el validation set.
    FIX: Los pasos de pipeline_ml se aplanan correctamente para que
         ImbPipeline no reciba un Pipeline anidado (causa del TypeError anterior).
    """
    if model_name is None:
        model_name = model.__class__.__name__

    # FIX: se aplanan los pasos de pipeline_ml para evitar el TypeError
    # "All intermediate steps of the chain should not be Pipelines"
    if use_smote:
        full_pipe = ImbPipeline(steps=[
            *pipeline_ml.steps,
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    else:
        full_pipe = Pipeline(steps=[
            *pipeline_ml.steps,
            ("model", model),
        ])

    # Validación cruzada — scoring con f1_macro para evaluar ambas clases
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # FIX: scoring macro-averaged para que el CV sea justo con clases desbalanceadas
    scoring_metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv_results = {}

    for metric in scoring_metrics:
        cv_results[metric] = cross_val_score(
            full_pipe, X_train, y_train, cv=kfold, scoring=metric
        )

    cv_df = pd.DataFrame(cv_results)

    # Fit final sobre todo X_train
    full_pipe.fit(X_train, y_train)
    y_pred = full_pipe.predict(X_test)

    # FIX: probabilidades para AUC real
    y_proba = None
    if hasattr(full_pipe, "predict_proba"):
        try:
            y_proba = full_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            pass

    # Train scores aproximados
    train_results = {
        metric: cross_val_score(full_pipe, X_train, y_train, cv=2, scoring=metric).mean()
        for metric in scoring_metrics
    }

    # Resumen de métricas
    metrics_summary = summarize_classification(
        y_test, y_pred, y_proba=y_proba, model_name=model_name
    )

    # Gráfico: Train vs CV
    metrics_df = pd.DataFrame({
        "Metric":      scoring_metrics,
        "Train Score": [train_results[m] for m in scoring_metrics],
        "CV Mean":     [cv_df[m].mean() for m in scoring_metrics],
        "CV Std":      [cv_df[m].std()  for m in scoring_metrics],
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cv_df.plot.box(ax=axes[0], title=f"CV Boxplot – {model_name}", ylabel="Score")
    metrics_df.plot(
        kind="bar", x="Metric", y=["Train Score", "CV Mean"],
        yerr="CV Std", ax=axes[1],
        title=f"Train vs CV – {model_name}",
        ylabel="Score", capsize=4, legend=True
    )
    plt.tight_layout()
    plt.show()

    # Gráfico: Curva de aprendizaje (recall macro para capturar clase 0)
    common_params = {
        "X": X_train, "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=30, test_size=0.2, random_state=123),
        "n_jobs": -1, "return_times": True,
    }
    tr_sz, tr_sc, te_sc, ft, _ = learning_curve(
        full_pipe, scoring="recall_macro", **common_params
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(tr_sz, tr_sc.mean(1), "o-", label="Train")
    axes[0].plot(tr_sz, te_sc.mean(1), "o-", color="orange", label="CV")
    axes[0].fill_between(tr_sz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.3)
    axes[0].fill_between(tr_sz, te_sc.mean(1)-te_sc.std(1), te_sc.mean(1)+te_sc.std(1), alpha=0.3, color="orange")
    axes[0].set_title(f"Consistencia – {model_name}")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Recall Macro")
    axes[0].legend()

    axes[1].plot(tr_sz, ft.mean(1), "o-")
    axes[1].fill_between(tr_sz, ft.mean(1)-ft.std(1), ft.mean(1)+ft.std(1), alpha=0.3)
    axes[1].set_title(f"Escalabilidad – {model_name}")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    plt.tight_layout()
    plt.show()

    # Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Matriz de Confusión – {model_name}")
    plt.show()

    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=["Mora (0)", "Paga (1)"]))

    return {
        "pipeline":      full_pipe,
        "y_pred":        y_pred,
        "y_proba":       y_proba,
        "cv_results":    cv_df,
        "summary":       metrics_summary,
        "train_results": train_results,
    }


# 2. Comparación de modelos

def compare_models(results: dict) -> pd.DataFrame:
    """
    Genera tabla comparativa y gráfico de barras entre modelos.
    Muestra métricas macro y específicas de la clase mora (0).
    """
    summaries = pd.concat(
        [v["summary"] for v in results.values()],
        ignore_index=True
    ).set_index("model")

    # Gráfico comparativo con métricas macro
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    summaries[["accuracy", "f1_macro", "roc_auc"]].plot(
        kind="bar", ax=axes[0], colormap="tab10", edgecolor="white"
    )
    axes[0].set_title("Métricas Generales (Macro)", fontweight="bold")
    axes[0].set_ylabel("Score")
    axes[0].set_xticklabels(summaries.index, rotation=30, ha="right")
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0, 1.1)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    # Gráfico específico para clase mora (la importante)
    summaries[["precision_mora", "recall_mora", "f1_mora"]].plot(
        kind="bar", ax=axes[1], colormap="Set2", edgecolor="white"
    )
    axes[1].set_title("Métricas Clase Mora (0) — La importante", fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].set_xticklabels(summaries.index, rotation=30, ha="right")
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(0, 1.1)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    plt.tight_layout()
    plt.show()

    return summaries.reset_index()


def select_best_model(results: dict, metric: str = "f1_mora") -> tuple:
    """
    Selecciona el mejor modelo según la métrica especificada.
    Por defecto usa f1_mora (clase 0) ya que en crédito detectar mora es lo crítico.
    """
    best_name = max(
        results,
        key=lambda k: float(results[k]["summary"][metric].values[0])
    )
    best = results[best_name]
    print(f"\n✅ Mejor modelo: {best_name} | {metric} = {best['summary'][metric].values[0]}")
    return best_name, best["pipeline"], best["summary"]


# 3. Ejecución directa

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ft_engineering import build_feature_pipeline

    print("=" * 55)
    print("   MODEL TRAINING – model_training.py")
    print("=" * 55)

    print("\nCargando y procesando datos...")
    df = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, pipeline_ml = build_feature_pipeline(df)

    print(f"\nBalance target train: {y_train.value_counts().to_dict()}")
    print(f"Clase minoritaria (mora): {y_train.value_counts(normalize=True)[0]:.1%} del total")
    print("→ Dataset MUY desbalanceado. Se comparan resultados con y sin SMOTE.\n")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # --- Con SMOTE (datos balanceados) ---
    print("\n" + "=" * 55)
    print("   CON SMOTE (clases balanceadas)")
    print("=" * 55)
    results_smote = {}
    for name, model in models.items():
        print(f"\nEntrenando: {name} [SMOTE]...")
        results_smote[name] = build_model(
            model, X_train, y_train, X_test, y_test,
            pipeline_ml=pipeline_ml,
            model_name=name,
            use_smote=True,
        )

    print("\nTabla comparativa — CON SMOTE:")
    tabla_smote = compare_models(results_smote)
    print(tabla_smote.to_string(index=False))

    # --- Sin SMOTE (datos desbalanceados) ---
    print("\n" + "=" * 55)
    print("   SIN SMOTE (clases desbalanceadas)")
    print("=" * 55)
    results_nosmote = {}
    for name, model in models.items():
        print(f"\nEntrenando: {name} [sin SMOTE]...")
        results_nosmote[name] = build_model(
            model, X_train, y_train, X_test, y_test,
            pipeline_ml=pipeline_ml,
            model_name=name,
            use_smote=False,
        )

    print("\nTabla comparativa — SIN SMOTE:")
    tabla_nosmote = compare_models(results_nosmote)
    print(tabla_nosmote.to_string(index=False))

    # Selección del mejor modelo (por f1 de la clase mora)
    print("\n" + "=" * 55)
    print("   SELECCIÓN DEL MEJOR MODELO")
    print("   Métrica de referencia: f1_mora (clase 0)")
    print("=" * 55)
    print("\nCon SMOTE:")
    best_name, best_pipeline, best_metrics = select_best_model(results_smote, metric="f1_mora")

    models_path = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(models_path, "best_model.pkl"))

    print(f"\nModelo guardado: {models_path}/best_model.pkl")
    print("\n" + "=" * 55)
    print("   Entrenamiento completado exitosamente.")
    print("=" * 55)