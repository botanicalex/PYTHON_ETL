from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, fbeta_score,
    classification_report, balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, cross_validate, learning_curve
from sklearn.base import clone
from xgboost import XGBClassifier

from ft_engineering import build_feature_pipeline, pipeline_ml

os.makedirs("outputs", exist_ok=True)

OUTPUT_DIR = Path("outputs")

# ─────────────────────────────────────────────
# Scorer principal: F-beta (beta=2, pos_label=0)
# Doble peso al Recall vs Precision
# → detectar morosos es más costoso que falsos positivos
# ─────────────────────────────────────────────

from sklearn.metrics import make_scorer
fbeta_mora   = make_scorer(fbeta_score, beta=2, pos_label=0, zero_division=0)
cv_strategy  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_cv = {
    "accuracy" : "accuracy",
    "f1"       : make_scorer(f1_score,       pos_label=0, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall"   : make_scorer(recall_score,    pos_label=0, zero_division=0),
    "fbeta"    : fbeta_mora,
}


# ─────────────────────────────────────────────
# 1. Dataclass de resultado
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    best_model_name    : str
    best_model_pipeline: ImbPipeline
    summary_table      : pd.DataFrame


# ─────────────────────────────────────────────
# 2. summarize_classification
# ─────────────────────────────────────────────

def summarize_classification(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy"          : accuracy_score(y_true, y_pred),
        "balanced_accuracy" : balanced_accuracy_score(y_true, y_pred),
        "precision"         : precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall"            : recall_score(y_true, y_pred,    pos_label=0, zero_division=0),
        "f1"                : f1_score(y_true, y_pred,        pos_label=0, zero_division=0),
        "fbeta"             : fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0),
        "roc_auc"           : roc_auc_score(y_true, y_pred),
        "casos_mora_pred"   : int(np.count_nonzero(y_pred == 0)),
    }


# ─────────────────────────────────────────────
# 3. build_model
# Pipeline: preprocesamiento + SMOTE + clasificador
# ─────────────────────────────────────────────

def build_model(estimator) -> ImbPipeline:
    return ImbPipeline(steps=[
        *clone(pipeline_ml).steps,
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", estimator),
    ])


# ─────────────────────────────────────────────
# 4. compute_selection_score
# Combina las 3 dimensiones con pesos explícitos:
# - Performance  (60%): F-beta en test
# - Consistency  (25%): estabilidad CV (menor std = mejor)
# - Scalability  (15%): velocidad de entrenamiento
# ─────────────────────────────────────────────

def compute_selection_score(row: pd.Series) -> float:
    performance  = row["test_fbeta"]
    consistency  = 1 / (1 + row["cv_fbeta_std"])
    scalability  = 1 / (1 + row["cv_fit_time_mean"])
    return 0.60 * performance + 0.25 * consistency + 0.15 * scalability


# ─────────────────────────────────────────────
# 5. evaluate_candidate
# Entrena y evalúa un modelo con CV + test
# ─────────────────────────────────────────────

def evaluate_candidate(
    model_name : str,
    estimator,
    X_train    : pd.DataFrame,
    X_test     : pd.DataFrame,
    y_train    : pd.Series,
    y_test     : pd.Series,
) -> Dict:
    model_name_clf = estimator.__class__.__name__
    pipe = build_model(estimator)

    # CV sin sample_weight (SMOTE dentro del pipeline balancea las clases)
    cv_output = cross_validate(
        build_model(clone(estimator)),
        X_train, y_train,
        cv=cv_strategy,
        scoring=scoring_cv,
        return_train_score=False,
        n_jobs=-1,
    )

    # Fit final — SMOTE dentro del pipeline balancea las clases
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    test   = summarize_classification(y_test, y_pred)

    # Curva de aprendizaje + escalabilidad
    plot_learning_curve(estimator, model_name, X_train, y_train)

    result = {
        "modelo"             : model_name,
        "cv_accuracy_mean"   : float(np.mean(cv_output["test_accuracy"])),
        "cv_precision_mean"  : float(np.mean(cv_output["test_precision"])),
        "cv_recall_mean"     : float(np.mean(cv_output["test_recall"])),
        "cv_f1_mean"         : float(np.mean(cv_output["test_f1"])),
        "cv_fbeta_mean"      : float(np.mean(cv_output["test_fbeta"])),
        "cv_fbeta_std"       : float(np.std(cv_output["test_fbeta"])),
        "cv_fit_time_mean"   : float(np.mean(cv_output["fit_time"])),
        "test_accuracy"      : test["accuracy"],
        "test_balanced_acc"  : test["balanced_accuracy"],
        "test_precision"     : test["precision"],
        "test_recall"        : test["recall"],
        "test_f1"            : test["f1"],
        "test_fbeta"         : test["fbeta"],
        "test_roc_auc"       : test["roc_auc"],
        "casos_mora_pred"    : test["casos_mora_pred"],
        "pipeline"           : pipe,
    }
    result["selection_score"] = compute_selection_score(pd.Series(result))
    return result


# ─────────────────────────────────────────────
# 6. plot_learning_curve
# Curva de aprendizaje + escalabilidad por modelo
# ─────────────────────────────────────────────

def plot_learning_curve(estimator, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    cv_lc = ShuffleSplit(n_splits=50, test_size=0.2, random_state=123)

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        build_model(clone(estimator)),
        X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv_lc,
        scoring=make_scorer(recall_score, pos_label=0, zero_division=0),
        n_jobs=-1,
        return_times=True,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    test_mean  = np.mean(test_scores,  axis=1)
    test_std   = np.std(test_scores,   axis=1)
    fit_mean   = np.mean(fit_times,    axis=1)
    fit_std    = np.std(fit_times,     axis=1)
    score_mean = np.mean(score_times,  axis=1)
    score_std  = np.std(score_times,   axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Curva de aprendizaje
    axes[0].plot(train_sizes, train_mean, "o-", label="Train")
    axes[0].plot(train_sizes, test_mean, "o-", color="orange", label="CV")
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    axes[0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange")
    axes[0].set_title(f"Curva de Aprendizaje\n{model_name}", fontweight="bold")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Recall (mora)")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Fit time
    axes[1].plot(train_sizes, fit_mean, "o-", color="steelblue")
    axes[1].fill_between(train_sizes, fit_mean - fit_std, fit_mean + fit_std, alpha=0.3)
    axes[1].set_title(f"Scalability — Fit time\n{model_name}", fontweight="bold")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].spines[["top", "right"]].set_visible(False)

    # Score time
    axes[2].plot(train_sizes, score_mean, "o-", color="seagreen")
    axes[2].fill_between(train_sizes, score_mean - score_std, score_mean + score_std, alpha=0.3, color="seagreen")
    axes[2].set_title(f"Scalability — Score time\n{model_name}", fontweight="bold")
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("Score time (s)")
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = OUTPUT_DIR / f"learning_curve_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Curva guardada: {path}")


# ─────────────────────────────────────────────
# 7. plot_model_comparison
# ─────────────────────────────────────────────

def plot_model_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data = summary.sort_values("selection_score", ascending=False)

    # Performance
    axes[0].bar(data["modelo"], data["test_fbeta"], color="steelblue", edgecolor="white")
    axes[0].set_title("Performance\n(Test F-beta mora)", fontweight="bold")
    axes[0].set_ylabel("F-beta")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Consistency
    axes[1].bar(data["modelo"], data["cv_fbeta_std"], color="coral", edgecolor="white")
    axes[1].set_title("Consistency\n(CV F-beta std — menor es mejor)", fontweight="bold")
    axes[1].set_ylabel("Std")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].spines[["top", "right"]].set_visible(False)

    # Scalability
    axes[2].bar(data["modelo"], data["cv_fit_time_mean"], color="seagreen", edgecolor="white")
    axes[2].set_title("Scalability\n(Fit time — menor es mejor)", fontweight="bold")
    axes[2].set_ylabel("Segundos")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Comparación de modelos — Performance · Consistency · Scalability",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparacion_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfico guardado: {OUTPUT_DIR / 'comparacion_modelos.png'}")


# ─────────────────────────────────────────────
# 7. train_and_select_model
# ─────────────────────────────────────────────

def train_and_select_model(data_path: str = "Base_de_datos.xlsx") -> ModelResult:
    df = pd.read_excel(data_path)
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    spw = round((y_train == 1).sum() / (y_train == 0).sum(), 2)
    print(f"\nscale_pos_weight (XGBoost): {spw}")
    print(f"Balance train: {y_train.value_counts(normalize=True).round(3).to_dict()}\n")

    candidatos = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42
        ),
    }

    resultados = []
    for nombre, clf in candidatos.items():
        print(f"\n{'='*55}")
        print(f"  Evaluando: {nombre}")
        print(f"{'='*55}")
        result = evaluate_candidate(nombre, clf, X_train, X_test, y_train, y_test)
        resultados.append(result)
        print(f"  CV F-beta mean : {result['cv_fbeta_mean']:.4f} ± {result['cv_fbeta_std']:.4f}")
        print(f"  Test Recall    : {result['test_recall']:.4f}")
        print(f"  Test F-beta    : {result['test_fbeta']:.4f}")
        print(f"  Test ROC-AUC   : {result['test_roc_auc']:.4f}")
        print(f"  Selection score: {result['selection_score']:.4f}")

    # Tabla resumen
    cols_tabla = [
        "modelo", "cv_fbeta_mean", "cv_fbeta_std", "cv_fit_time_mean",
        "test_recall", "test_f1", "test_fbeta", "test_roc_auc",
        "casos_mora_pred", "selection_score",
    ]
    summary = pd.DataFrame(resultados)[cols_tabla].sort_values(
        "selection_score", ascending=False
    ).reset_index(drop=True)

    print("\n=== TABLA RESUMEN ===")
    print(summary.drop(columns=[]).round(4).to_string(index=False))

    # Gráfico comparativo
    plot_model_comparison(summary)

    # Mejor modelo
    mejor_idx    = summary.iloc[0]["modelo"]
    mejor_pipe   = next(r["pipeline"] for r in resultados if r["modelo"] == mejor_idx)

    print(f"\n=== MEJOR MODELO: {mejor_idx} ===")
    y_pred_final = mejor_pipe.predict(X_test)
    print(classification_report(
        y_test, y_pred_final,
        target_names=["Mora", "Paga a tiempo"],
        zero_division=0,
    ))

    joblib.dump(mejor_pipe, "mejor_modelo.pkl")
    print("Modelo guardado: mejor_modelo.pkl")

    return ModelResult(
        best_model_name     = mejor_idx,
        best_model_pipeline = mejor_pipe,
        summary_table       = summary,
    )


# ─────────────────────────────────────────────
# 8. Ejecución principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("   MODEL TRAINING – model_training.py")
    print("=" * 55)

    result = train_and_select_model()

    print(f"\nMejor modelo     : {result.best_model_name}")
    print(f"Selection score  : {result.summary_table.iloc[0]['selection_score']:.4f}")