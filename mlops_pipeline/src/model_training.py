import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold, ShuffleSplit,
    cross_validate, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier

from ft_engineering import build_feature_pipeline, pipeline_ml


# ─────────────────────────────────────────────
# Scorers apuntando a clase 0 (mora)
# pos_label=0 es clave dado el desbalance 95/5%
# ─────────────────────────────────────────────

scoring_cv = {
    "accuracy" : "accuracy",
    "f1"       : make_scorer(f1_score,        pos_label=0, zero_division=0),
    "precision": make_scorer(precision_score,  pos_label=0, zero_division=0),
    "recall"   : make_scorer(recall_score,     pos_label=0, zero_division=0),
}


# ─────────────────────────────────────────────
# summarize_classification
# ─────────────────────────────────────────────

def summarize_classification(y_test, y_pred):
    return {
        "accuracy"           : accuracy_score(y_test, y_pred, normalize=True),
        "precision"          : precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall"             : recall_score(y_test, y_pred,    pos_label=0, zero_division=0),
        "f1_score"           : f1_score(y_test, y_pred,        pos_label=0, zero_division=0),
        "roc_auc"            : roc_auc_score(y_test, y_pred),
        "casosNoPagoAtiempo" : int(np.count_nonzero(y_pred == 0)),
    }


# ─────────────────────────────────────────────
# build_model
# ─────────────────────────────────────────────

def build_model(classifier_fn, data_params: dict, test_frac: float = 0.2) -> dict:
    """
    Entrena un modelo de clasificación y evalúa:
    - Performance : métricas train/test
    - Consistency : cross-validation (StratifiedKFold, pos_label=0)
    - Scalability : learning curve (fit time / score time)

    Args:
        classifier_fn : clasificador sklearn-compatible
        data_params   : dict con 'name_of_y_col', 'names_of_x_cols', 'dataset'
        test_frac     : fracción de datos para test (default 0.2)

    Returns:
        dict con métricas de train y test
    """

    name_of_y_col   = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset         = data_params["dataset"]

    X = dataset[names_of_x_cols]
    y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=1234
    )

    # Pipeline completo: preprocesamiento + clasificador
    classifier_pipe = Pipeline(steps=[
        *clone(pipeline_ml).steps,
        ("model", classifier_fn),
    ])
    model = classifier_pipe.fit(x_train, y_train)

    y_pred       = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary  = summarize_classification(y_test,  y_pred)

    model_name = model.steps[-1][1].__class__.__name__

    # ── Consistency: StratifiedKFold CV ───────
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        Pipeline(steps=[*clone(pipeline_ml).steps, ("model", clone(classifier_fn))]),
        x_train, y_train,
        cv=kfold,
        scoring=scoring_cv,
    )

    cv_results_df = pd.DataFrame({
        k.replace("test_", ""): v
        for k, v in scores.items()
        if k.startswith("test_")
    })

    print(f"\n  CV results ({model_name}) — clase mora (pos_label=0):")
    print(cv_results_df.mean().round(4).to_string())

    # ── Scalability: Learning Curve ────────────
    common_params = {
        "X"           : x_train,
        "y"           : y_train,
        "train_sizes" : np.linspace(0.1, 1.0, 5),
        "cv"          : ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        "n_jobs"      : -1,
        "return_times": True,
    }

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        Pipeline(steps=[*clone(pipeline_ml).steps, ("model", clone(classifier_fn))]),
        scoring=make_scorer(recall_score, pos_label=0, zero_division=0),
        **common_params,
    )

    train_mean       = np.mean(train_scores, axis=1)
    train_std        = np.std(train_scores,  axis=1)
    test_mean        = np.mean(test_scores,  axis=1)
    test_std         = np.std(test_scores,   axis=1)
    fit_times_mean   = np.mean(fit_times,    axis=1)
    fit_times_std    = np.std(fit_times,     axis=1)
    score_times_mean = np.mean(score_times,  axis=1)
    score_times_std  = np.std(score_times,   axis=1)

    # Curva de aprendizaje
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, test_mean,  "o-", color="orange", label="Cross-validation score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    ax.fill_between(train_sizes, test_mean  - test_std,  test_mean  + test_std,  alpha=0.3, color="orange")
    ax.set_title(f"Learning Curve – {model_name}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Recall (mora)")
    ax.legend(loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    # Escalabilidad
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    ax[0].plot(train_sizes, fit_times_mean, "o-")
    ax[0].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.3)
    ax[0].set_ylabel("Fit time (s)")
    ax[0].set_title(f"Scalability – {model_name}")
    ax[1].plot(train_sizes, score_times_mean, "o-")
    ax[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha=0.3)
    ax[1].set_ylabel("Score time (s)")
    ax[1].set_xlabel("Number of training samples")
    plt.tight_layout()
    plt.show()

    print("\n  Training Sizes      :", train_sizes)
    print("  Train Recall Mean   :", train_mean.round(4))
    print("  CV Recall Mean      :", test_mean.round(4))
    print("  Fit Times Mean (s)  :", fit_times_mean.round(4))
    print("  Score Times Mean (s):", score_times_mean.round(4))

    return {"train": train_summary, "test": test_summary}


# ─────────────────────────────────────────────
# Ejecución principal
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("   MODEL TRAINING – model_training.py")
    print("=" * 55)

    df = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    # Reconstruir dataset para pasarlo a build_model
    df_model = pd.concat([X_train, X_test])
    df_model["Pago_atiempo"] = pd.concat([y_train, y_test])

    data_params = {
        "name_of_y_col"  : "Pago_atiempo",
        "names_of_x_cols": X_train.columns.tolist(),
        "dataset"        : df_model,
    }

    spw = round((y_train == 1).sum() / (y_train == 0).sum(), 2)
    print(f"scale_pos_weight (XGBoost): {spw}\n")

    modelos = {
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

    # ── Entrenar y comparar ───────────────────
    resultados = []

    for nombre, clf in modelos.items():
        print(f"\n{'='*55}")
        print(f"  Modelo: {nombre}")
        print(f"{'='*55}")
        result = build_model(clf, data_params)

        resultados.append({
            "Modelo"             : nombre,
            "Train Accuracy"     : round(result["train"]["accuracy"],  4),
            "Train F1 (mora)"    : round(result["train"]["f1_score"],  4),
            "Train Recall (mora)": round(result["train"]["recall"],    4),
            "Test Accuracy"      : round(result["test"]["accuracy"],   4),
            "Test F1 (mora)"     : round(result["test"]["f1_score"],   4),
            "Test Recall (mora)" : round(result["test"]["recall"],     4),
            "Test ROC-AUC"       : round(result["test"]["roc_auc"],    4),
            "Casos mora pred."   : result["test"]["casosNoPagoAtiempo"],
        })

    # ── Tabla resumen ─────────────────────────
    df_resultados = pd.DataFrame(resultados).set_index("Modelo")
    print("\n=== TABLA RESUMEN ===")
    print(df_resultados.to_string())

    # ── Gráfico comparativo ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_resultados[["Test F1 (mora)", "Test Recall (mora)", "Test ROC-AUC"]].plot(
        kind="bar", ax=axes[0], rot=15, colormap="coolwarm", edgecolor="white"
    )
    axes[0].set_title("Comparación de métricas en test", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].spines[["top", "right"]].set_visible(False)

    df_resultados[["Train F1 (mora)", "Test F1 (mora)"]].plot(
        kind="bar", ax=axes[1], rot=15, colormap="viridis", edgecolor="white"
    )
    axes[1].set_title("F1 mora: Train vs Test", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=150)
    plt.show()
    print("\nGráfico guardado: comparacion_modelos.png")

    # ── Mejor modelo y guardado ───────────────
    mejor_nombre = df_resultados["Test F1 (mora)"].idxmax()
    print(f"\n=== MEJOR MODELO: {mejor_nombre} ===")

    mejor_pipe = Pipeline(steps=[
        *clone(pipeline_ml).steps,
        ("model", modelos[mejor_nombre]),
    ])
    mejor_pipe.fit(X_train, y_train)

    print(classification_report(
        y_test, mejor_pipe.predict(X_test),
        target_names=["Mora", "Paga a tiempo"]
    ))

    joblib.dump(mejor_pipe, "mejor_modelo.pkl")
    print("Modelo guardado: mejor_modelo.pkl")