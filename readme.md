# PYTHON_ETL - Predicción de Mora Crediticia

**Ciencia de Datos en Producción - Entregable 4**  
**Docente:** Juan Sebastián Parra Sánchez  
**Autora:** Alexandra Vasco Lopera

## Descripción del Proyecto

Pipeline MLOps completo para predicción de incumplimiento crediticio en una empresa financiera. El objetivo es predecir si un cliente pagará a tiempo su crédito (`Pago_atiempo = 1`) o entrará en mora (`Pago_atiempo = 0`), usando aprendizaje automático con despliegue en API REST.

Los datos se leen directamente desde **GCP Storage** (archivo Excel en bucket) y se cargan a **BigQuery** para análisis y entrenamiento en la nube. El pipeline completo está orquestado con **Vertex AI Pipelines** y la integración continua se gestiona con **Jenkins** corriendo en Docker.

- **Dataset:** 10,763 registros | 23 variables
- **Variable objetivo:** `Pago_atiempo` (desbalance 95% / 5%)
- **Mejor modelo local:** Logistic Regression con SMOTE (Recall mora: 0.65 | ROC-AUC: 0.7005)
- **Mejor modelo cloud:** AutoML GCP (Recall: 0.953 | ROC-AUC: 0.967)

## Estructura del Proyecto

```
PYTHON_ETL/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb          # Carga y exploración inicial del dataset
│       ├── comprension_eda.ipynb       # Análisis exploratorio de datos (EDA)
│       ├── ft_engineering.py           # Pipeline de feature engineering
│       ├── gcp_utils.py                # Descarga de datos desde GCP Storage
│       ├── heuristic_model.py          # Modelo heurístico baseline
│       ├── model_training.py           # Entrenamiento y comparación de modelos
│       ├── model_deploy.py             # API REST con FastAPI
│       ├── model_evaluation.py         # Dashboard de métricas del modelo
│       ├── model_monitoring.py         # Detección de Data Drift (KS test)
│       ├── upload_to_bigquery.py       # Carga del dataset a BigQuery
│       ├── vertex_pipeline.py          # Pipeline KFP de 4 componentes en Vertex AI
│       ├── model_comparison.py         # Comparación modelo local vs AutoML GCP
│       ├── model_stability.py          # Validación cruzada y análisis de estabilidad
│       ├── mejor_modelo.pkl            # Modelo entrenado serializado
│       └── outputs/                    # Gráficos generados por los módulos
│           ├── learning_curve_*.png    # Curvas de aprendizaje por modelo
│           ├── comparacion_modelos.png # Comparación Performance·Consistency·Scalability
│           ├── model_evaluation.png    # Dashboard de métricas del modelo desplegado
│           └── model_monitoring.png    # Reporte visual de monitoreo y drift
├── trainer/                            # Paquete de custom training para Vertex AI
│   ├── __init__.py
│   └── task.py                         # Script de entrenamiento personalizado
├── Dockerfile                          # Imagen Docker para despliegue de la API
├── Jenkinsfile                         # Pipeline CI/CD para Jenkins
├── setup.py                            # Empaquetado del trainer para Vertex AI
├── requirements.txt                    # Dependencias del proyecto
├── config.json                         # Configuración del proyecto (incluye GCP)
├── .gitignore                          # Exclusiones para Git
└── set_up.bat                          # Script de configuración inicial
```

## Infraestructura Cloud (Entregable 4)

| Servicio | Recurso |
|----------|---------|
| GCP Storage | `gs://credit-risk-mlops-alex-data/` — almacena el dataset Excel, modelos y pipeline artifacts |
| BigQuery | `credit-risk-mlops-alex.credit_risk_dataset_east.creditos_raw` — tabla con 10,763 registros en región `us-east1` |
| Vertex AI AutoML | Modelo `credit-risk-dataset-east` entrenado con los datos de BigQuery (ROC-AUC: 0.967) |
| Vertex AI Pipelines | Pipeline KFP de 4 componentes: carga → preprocesamiento → entrenamiento → evaluación |
| Custom Training | Paquete `credit_risk_trainer-1.0.0.tar.gz` subido a `gs://credit-risk-mlops-alex-data/trainer/` |
| Jenkins | Pipeline CI/CD corriendo localmente vía Docker en `localhost:8080` |

### Credenciales GCP

El archivo `gcp_credentials.json` (service account `credit-risk-sa@credit-risk-mlops-alex.iam.gserviceaccount.com`) debe colocarse en la raíz del proyecto. **No se versiona** (incluido en `.gitignore`).

## Pipeline CI/CD

El archivo `Jenkinsfile` define un pipeline de integración continua con 4 stages que se activa automáticamente en cada push a `master`:

| Stage | Descripción |
|-------|-------------|
| Clonar repositorio | Checkout desde `https://github.com/botanicalex/PYTHON_ETL` |
| Verificar estructura | Valida existencia de los 8 archivos clave del proyecto |
| Validar scripts Python | Compila sintaxis de los 5 módulos `.py` con `py_compile` |
| Validar config.json | Verifica que todas las claves requeridas estén presentes |

Al finalizar envía notificación por email a `alexavascolopera@gmail.com`.

### Levantar Jenkins con Docker

```bash
# Construir imagen personalizada con Python 3
docker-compose up -d

# Acceder a Jenkins
http://localhost:8080
```

## Instalación

### Prerrequisitos
- Python >= 3.10
- Git
- Docker Desktop (para Jenkins y despliegue de la API)
- Cuenta GCP con acceso al proyecto `credit-risk-mlops-alex`

### Pasos de instalación

1. Clona el repositorio:
```bash
git clone https://github.com/botanicalex/PYTHON_ETL.git
cd PYTHON_ETL
```

2. Ejecuta el script de configuración:
```bash
set_up.bat
```

3. Coloca el archivo de credenciales GCP en la raíz:
```
gcp_credentials.json   ← service account key (no versionar)
```

### Instalación manual (alternativa)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### 1. Feature Engineering (lee desde GCP Storage)
```bash
cd mlops_pipeline/src
python ft_engineering.py
```

### 2. Cargar datos a BigQuery
```bash
python upload_to_bigquery.py
```
Sube `Base_de_datos.xlsx` desde GCS a `credit_risk_dataset_east.creditos_raw`.

### 3. Modelo Heurístico (baseline)
```bash
python heuristic_model.py
```

### 4. Entrenamiento de modelos
```bash
python model_training.py
```
Genera `mejor_modelo.pkl` y gráficos comparativos.

### 5. Comparación local vs AutoML GCP
```bash
python model_comparison.py
```
Genera `comparison_bar.png` y `comparison_radar.png`.

### 6. Análisis de estabilidad
```bash
python model_stability.py
```
Validación cruzada 5 folds. Genera `stability_boxplot.png`.

### 7. Vertex AI Pipeline
```bash
python vertex_pipeline.py
```
Compila y envía el pipeline KFP a Vertex AI Pipelines.

### 8. Despliegue de la API
```bash
python model_deploy.py
```
API disponible en `http://localhost:8000`  
Documentación interactiva en `http://localhost:8000/docs`

### 9. Con Docker
```bash
docker build -t modelo-mora .
docker run -p 8000:8000 modelo-mora
```

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Health check básico |
| GET | `/health` | Estado del modelo |
| POST | `/predict` | Predicción por batch |
| GET | `/evaluation` | Dashboard de métricas (PNG) |
| POST | `/monitor` | Detección de Data Drift |

## Pipeline MLOps

```
GCS (Excel) → BigQuery → Feature Engineering → Split → Entrenamiento → Evaluación → Despliegue → Monitoreo
```

### Modelos evaluados

| Modelo | CV F-beta | Test Recall | Test ROC-AUC |
|--------|-----------|-------------|--------------|
| Logistic Regression | 0.2573 | 0.65 | 0.7005 |
| Random Forest | 0.0183 | 0.03 | 0.6677 |
| Gradient Boosting | 0.0501 | 0.07 | 0.6700 |
| XGBoost | 0.0093 | 0.02 | 0.6548 |
| **AutoML GCP** | — | **0.953** | **0.967** |

**Modelo local seleccionado:** Logistic Regression (class_weight=balanced, SMOTE)  
**Criterio de selección:** Selection Score = Performance (60% F-beta) + Consistency (25% estabilidad CV) + Scalability (15% velocidad)

**Modelo cloud:** AutoML GCP supera al modelo local en Recall (+46%) y ROC-AUC (+37%).

## Tecnologías Utilizadas

- **Python >= 3.10**
- **scikit-learn** — modelos y pipelines
- **XGBoost** — gradient boosting
- **imbalanced-learn** — SMOTE para desbalance de clases
- **FastAPI + Uvicorn** — API REST
- **joblib** — serialización del modelo
- **pandas / numpy** — procesamiento de datos
- **matplotlib / seaborn** — visualizaciones
- **scipy** — test KS para detección de drift
- **Docker** — contenedorización
- **Google Cloud Storage** — almacenamiento de datos y modelos
- **BigQuery** — data warehouse para entrenamiento en la nube
- **Vertex AI** — AutoML, Pipelines y Custom Training
- **KFP (Kubeflow Pipelines)** — orquestación de pipelines ML
- **Jenkins** — integración continua CI/CD

## Autor

**Alexandra Vasco Lopera**  
- GitHub: [@botanicalex](https://github.com/botanicalex)
- Email: alexavascolopera@gmail.com

## Licencia

Este proyecto es parte de un entregable académico — Ciencia de Datos en Producción 2026-1.
