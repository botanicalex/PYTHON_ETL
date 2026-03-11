import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    make_scorer, fbeta_score,
)
from sklearn.model_selection import (
    StratifiedKFold, ShuffleSplit,
    cross_validate, learning_curve, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ft_engineering import build_feature_pipeline, pipeline_ml


# ─────────────────────────────────────────────
# Scorers
# pos_label=0 → clase mora (minoritaria, 5%)
# fbeta_mora (beta=2): doble peso al Recall vs Precision
#   → detectar morosos es más costoso que falsos positivos
# ─────────────────────────────────────────────

scoring_cv = {
    "accuracy" : "accuracy",
    "f1"       : make_scorer(f1_score,        pos_label=0, zero_division=0),
    "precision": make_scorer(precision_score,  pos_label=0, zero_division=0),
    "recall"   : make_scorer(recall_score,     pos_label=0, zero_division=0),
}

fbeta_mora  = make_scorer(fbeta_score, beta=2, pos_label=0, zero_division=0)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


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
# STAGE 1 — build_model
# Entrena y evalúa modelos base (sin tuning).
# Evalúa Performance, Consistency y Scalability.
# ─────────────────────────────────────────────

def build_model(classifier_fn, X_train, X_test, y_train, y_test) -> dict:
    """
    Entrena un modelo base y evalúa tres dimensiones:
    - Performance  : métricas train/test
    - Consistency  : StratifiedKFold CV (pos_label=0)
    - Scalability  : learning curve (recall mora vs tamaño)

    Returns dict con métricas de train, test y CV.
    """
    model_name = classifier_fn.__class__.__name__

    # Pipeline: preprocesamiento + clasificador
    classifier_pipe = Pipeline(steps=[
        *clone(pipeline_ml).steps,
        ("model", classifier_fn),
    ])

    # GradientBoosting no soporta class_weight → sample_weight manual
    if model_name == "GradientBoostingClassifier":
        sample_w = compute_sample_weight(class_weight="balanced", y=y_train)
        model = classifier_pipe.fit(X_train, y_train, model__sample_weight=sample_w)
    else:
        model = classifier_pipe.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary  = summarize_classification(y_test,  y_pred)

    # ── Consistency: StratifiedKFold CV ───────
    cv_pipe = Pipeline(steps=[
        *clone(pipeline_ml).steps,
        ("model", clone(classifier_fn)),
    ])
    scores = cross_validate(cv_pipe, X_train, y_train, cv=cv_strategy, scoring=scoring_cv)
    cv_results_df = pd.DataFrame({
        k.replace("test_", ""): v
        for k, v in scores.items()
        if k.startswith("test_")
    })
    cv_mean = cv_results_df.mean()

    print(f"\n  CV results ({model_name}) — clase mora (pos_label=0):")
    print(cv_mean.round(4).to_string())

    # ── Scalability: Learning Curve ────────────
    lc_pipe = Pipeline(steps=[
        *clone(pipeline_ml).steps,
        ("model", clone(classifier_fn)),
    ])
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        lc_pipe,
        X=X_train, y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        scoring=make_scorer(recall_score, pos_label=0, zero_division=0),
        n_jobs=-1,
        return_times=True,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    test_mean  = np.mean(test_scores,  axis=1)
    test_std   = np.std(test_scores,   axis=1)
    fit_times_mean  = np.mean(fit_times,   axis=1)
    fit_times_std   = np.std(fit_times,    axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std  = np.std(score_times,  axis=1)

    # Curva de aprendizaje
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, test_mean,  "o-", color="orange", label="CV score")
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

    return {
        "train"   : train_summary,
        "test"    : test_summary,
        "cv_mean" : cv_mean.to_dict(),
    }


# ─────────────────────────────────────────────
# STAGE 2 — tune_model
# SMOTE + GridSearchCV optimizando F-beta (beta=2).
# SMOTE dentro del pipeline para evitar leakage en CV.
# ─────────────────────────────────────────────

param_grids = {
    "Logistic Regression": {
        "model__C"           : [0.01, 0.1, 1, 10],
        "model__solver"      : ["saga"],
        "model__max_iter": [2000],
        "model__class_weight": ["balanced"],
    },
    "Random Forest": {
        "model__n_estimators"    : [100, 200],
        "model__max_depth"       : [3, 5, 10],
        "model__min_samples_leaf": [5, 10, 20],
        "model__class_weight"    : ["balanced"],
    },
    "Gradient Boosting": {
        "model__n_estimators" : [50, 100, 200],
        "model__max_depth"    : [2, 3, 5],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample"    : [0.7, 1.0],
    },
    "XGBoost": {
        "model__n_estimators"    : [100, 200],
        "model__max_depth"       : [3, 5, 7],
        "model__learning_rate"   : [0.01, 0.05, 0.1],
        "model__subsample"       : [0.7, 1.0],
        "model__colsample_bytree": [0.7, 1.0],
    },
}


def tune_model(nombre, clf, param_grid, X_train, X_test, y_train, y_test) -> dict:
    """
    Aplica SMOTE + GridSearchCV optimizando F-beta (beta=2) sobre clase mora.

    Pipeline interno:
        preprocesamiento → SMOTE → clasificador

    Returns dict con métricas y el mejor estimador encontrado.
    """
    print(f"\n{'='*55}")
    print(f"  Tuning: {nombre}")
    print(f"{'='*55}")

    pipe = ImbPipeline(steps=[
        *clone(pipeline_ml).steps,
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", clf),
    ])

    grid_search = GridSearchCV(
        estimator  = pipe,
        param_grid = param_grid,
        scoring    = fbeta_mora,
        cv         = cv_strategy,
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    grid_search.fit(X_train, y_train)

    best_cv_score = grid_search.best_score_
    y_pred        = grid_search.predict(X_test)

    recall_t    = recall_score(y_test,    y_pred, pos_label=0, zero_division=0)
    f1_t        = f1_score(y_test,        y_pred, pos_label=0, zero_division=0)
    precision_t = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    fbeta_t     = fbeta_score(y_test,     y_pred, beta=2, pos_label=0, zero_division=0)
    roc_t       = roc_auc_score(y_test,   y_pred)

    print(f"\n  Mejores hiperparámetros:")
    for k, v in grid_search.best_params_.items():
        print(f"    {k}: {v}")

    print(f"\n  CV F-beta Mean (mejor): {best_cv_score:.4f}")
    print(f"\n  Métricas en test:")
    print(f"    Recall (mora)   : {recall_t:.4f}")
    print(f"    Precision (mora): {precision_t:.4f}")
    print(f"    F1 (mora)       : {f1_t:.4f}")
    print(f"    F-beta (mora)   : {fbeta_t:.4f}")
    print(f"    ROC-AUC         : {roc_t:.4f}")

    print(f"\n  Reporte completo:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Mora", "Paga a tiempo"],
        zero_division=0,
    ))

    return {
        "nombre"        : nombre,
        "best_params"   : grid_search.best_params_,
        "CV Fbeta Mean" : round(best_cv_score, 4),
        "Test Recall"   : round(recall_t,      4),
        "Test Precision": round(precision_t,   4),
        "Test F1"       : round(f1_t,          4),
        "Test Fbeta"    : round(fbeta_t,       4),
        "Test ROC-AUC"  : round(roc_t,         4),
        "best_estimator": grid_search.best_estimator_,
    }


# ─────────────────────────────────────────────
# Ejecución principal
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("   MODEL TRAINING – model_training.py")
    print("=" * 55)

    df = pd.read_excel("Base_de_datos.xlsx")
    X_train, X_test, y_train, y_test, _ = build_feature_pipeline(df)

    spw = round((y_train == 1).sum() / (y_train == 0).sum(), 2)
    print(f"\nscale_pos_weight (XGBoost): {spw}")
    print(f"Balance train: {y_train.value_counts(normalize=True).round(3).to_dict()}\n")

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

    # ─────────────────────────────────────────
    # STAGE 1 — Modelos base
    # ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   STAGE 1 — MODELOS BASE")
    print("=" * 55)

    resultados_base = []
    for nombre, clf in modelos.items():
        print(f"\n{'='*55}")
        print(f"  Modelo: {nombre}")
        print(f"{'='*55}")
        result = build_model(clf, X_train, X_test, y_train, y_test)
        resultados_base.append({
            "Modelo"             : nombre,
            "Train Recall (mora)": round(result["train"]["recall"],          4),
            "Train F1 (mora)"    : round(result["train"]["f1_score"],        4),
            "Test Recall (mora)" : round(result["test"]["recall"],           4),
            "Test F1 (mora)"     : round(result["test"]["f1_score"],         4),
            "Test ROC-AUC"       : round(result["test"]["roc_auc"],          4),
            "CV Recall Mean"     : round(result["cv_mean"].get("recall", 0), 4),
            "CV F1 Mean"         : round(result["cv_mean"].get("f1", 0),     4),
            "Casos mora pred."   : result["test"]["casosNoPagoAtiempo"],
        })

    df_base = pd.DataFrame(resultados_base).set_index("Modelo")
    print("\n=== TABLA RESUMEN — MODELOS BASE ===")
    print(df_base.to_string())

    # Gráfico modelos base
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_base[["Test F1 (mora)", "Test Recall (mora)", "Test ROC-AUC"]].plot(
        kind="bar", ax=axes[0], rot=15, colormap="coolwarm", edgecolor="white"
    )
    axes[0].set_title("Métricas en test", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].spines[["top", "right"]].set_visible(False)

    df_base[["Train F1 (mora)", "Test F1 (mora)"]].plot(
        kind="bar", ax=axes[1], rot=15, colormap="viridis", edgecolor="white"
    )
    axes[1].set_title("F1 mora: Train vs Test", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)

    df_base[["CV Recall Mean", "Test Recall (mora)"]].plot(
        kind="bar", ax=axes[2], rot=15, colormap="plasma", edgecolor="white"
    )
    axes[2].set_title("Recall mora: CV Mean vs Test", fontweight="bold")
    axes[2].set_ylim(0, 1)
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("comparacion_modelos_base.png", dpi=150)
    plt.show()
    print("\nGráfico guardado: comparacion_modelos_base.png")

    # ─────────────────────────────────────────
    # STAGE 2 — Tuning con SMOTE + GridSearchCV
    # ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   STAGE 2 — SMOTE + GRIDSEARCHCV (F-beta, beta=2)")
    print("=" * 55)

    clasificadores_tuning = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest"      : RandomForestClassifier(random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(random_state=42),
        "XGBoost"            : XGBClassifier(eval_metric="logloss", random_state=42),
    }

    resultados_tuning = []
    for nombre, clf in clasificadores_tuning.items():
        resultado = tune_model(
            nombre, clf,
            param_grids[nombre],
            X_train, X_test,
            y_train, y_test,
        )
        resultados_tuning.append(resultado)

    df_tuning = pd.DataFrame([{
        "Modelo"        : r["nombre"],
        "CV Fbeta Mean" : r["CV Fbeta Mean"],
        "Test Recall"   : r["Test Recall"],
        "Test Precision": r["Test Precision"],
        "Test F1"       : r["Test F1"],
        "Test Fbeta"    : r["Test Fbeta"],
        "Test ROC-AUC"  : r["Test ROC-AUC"],
    } for r in resultados_tuning]).set_index("Modelo")

    print("\n=== TABLA RESUMEN — MODELOS CON TUNING ===")
    print(df_tuning.to_string())

    # Gráfico tuning
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_tuning[["CV Fbeta Mean", "Test Fbeta"]].plot(
        kind="bar", ax=axes[0], rot=15, colormap="coolwarm", edgecolor="white"
    )
    axes[0].set_title("F-beta mora: CV Mean vs Test\n(SMOTE + tuning)", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].spines[["top", "right"]].set_visible(False)

    df_tuning[["Test F1", "Test Precision", "Test Recall"]].plot(
        kind="bar", ax=axes[1], rot=15, colormap="viridis", edgecolor="white"
    )
    axes[1].set_title("Métricas en test\n(SMOTE + tuning)", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("comparacion_modelos_tuning.png", dpi=150)
    plt.show()
    print("\nGráfico guardado: comparacion_modelos_tuning.png")

    # ─────────────────────────────────────────
    # SELECCIÓN FINAL
    # Criterio: mayor CV F-beta Mean (beta=2)
    # → mejor balance Recall/Precision con más peso al Recall
    # ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   SELECCIÓN FINAL DEL MEJOR MODELO")
    print("=" * 55)

    mejor_idx    = df_tuning["CV Fbeta Mean"].idxmax()
    mejor_result = next(r for r in resultados_tuning if r["nombre"] == mejor_idx)

    print(f"\n  Mejor modelo     : {mejor_idx}")
    print(f"  CV F-beta Mean   : {mejor_result['CV Fbeta Mean']}")
    print(f"  Test Recall      : {mejor_result['Test Recall']}")
    print(f"  Test Precision   : {mejor_result['Test Precision']}")
    print(f"  Test F1          : {mejor_result['Test F1']}")
    print(f"  Test F-beta      : {mejor_result['Test Fbeta']}")
    print(f"  Test ROC-AUC     : {mejor_result['Test ROC-AUC']}")

    print(f"\n  Reporte final en test:")
    y_pred_final = mejor_result["best_estimator"].predict(X_test)
    print(classification_report(
        y_test, y_pred_final,
        target_names=["Mora", "Paga a tiempo"],
        zero_division=0,
    ))

    joblib.dump(mejor_result["best_estimator"], "mejor_modelo.pkl")
    print(f"\n✅ Modelo guardado: mejor_modelo.pkl ({mejor_idx})")