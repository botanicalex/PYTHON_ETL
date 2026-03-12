# PYTHON_ETL - Predicción de Mora Crediticia

**Ciencia de Datos en Producción - Entregable 3**  
**Docente:** Juan Sebastián Parra Sánchez  
**Autora:** Alexandra Vasco Lopera

## Descripción del Proyecto

Pipeline MLOps completo para predicción de incumplimiento crediticio en una empresa financiera. El objetivo es predecir si un cliente pagará a tiempo su crédito (`Pago_atiempo = 1`) o entrará en mora (`Pago_atiempo = 0`), usando aprendizaje automático con despliegue en API REST.

- **Dataset:** 10,763 registros | 23 variables
- **Variable objetivo:** `Pago_atiempo` (desbalance 95% / 5%)
- **Mejor modelo:** Logistic Regression con SMOTE (Recall mora: 0.65 | ROC-AUC: 0.6493)

## Estructura del Proyecto

```
PYTHON_ETL/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb          # Carga y exploración inicial del dataset
│       ├── comprension_eda.ipynb       # Análisis exploratorio de datos (EDA)
│       ├── ft_engineering.py           # Pipeline de feature engineering
│       ├── heuristic_model.py          # Modelo heurístico baseline
│       ├── model_training.py           # Entrenamiento y comparación de modelos
│       ├── model_deploy.py             # API REST con FastAPI
│       ├── model_evaluation.py         # Dashboard de métricas del modelo
│       ├── model_monitoring.py         # Detección de Data Drift (KS test)
│       ├── mejor_modelo.pkl            # Modelo entrenado serializado
│       └── outputs/                    # Gráficos generados por los módulos
│           ├── learning_curve_*.png    # Curvas de aprendizaje por modelo
│           ├── comparacion_modelos.png # Comparación Performance·Consistency·Scalability
│           ├── model_evaluation.png    # Dashboard de métricas del modelo desplegado
│           └── model_monitoring.png    # Reporte visual de monitoreo y drift
├── Dockerfile                          # Imagen Docker para despliegue
├── requirements.txt                    # Dependencias del proyecto
├── config.json                         # Configuración del proyecto
├── .gitignore                          # Exclusiones para Git
└── set_up.bat                          # Script de configuración inicial
```

## Instalación

### Prerrequisitos
- Python >= 3.10
- Git
- Docker Desktop (para despliegue en contenedor)

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

Este script crea el entorno virtual e instala todas las dependencias.

### Instalación manual (alternativa)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### 1. Feature Engineering
```bash
cd mlops_pipeline/src
python ft_engineering.py
```

### 2. Modelo Heurístico (baseline)
```bash
python heuristic_model.py
```

### 3. Entrenamiento de modelos
```bash
python model_training.py
```
Genera `mejor_modelo.pkl` y gráficos comparativos.

### 4. Despliegue de la API
```bash
python model_deploy.py
```
API disponible en `http://localhost:8000`  
Documentación interactiva en `http://localhost:8000/docs`

### 5. Con Docker
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
Datos → Feature Engineering → Split → Entrenamiento → Evaluación → Despliegue → Monitoreo
```

### Modelos evaluados
| Modelo | CV F-beta | Test Recall | Test ROC-AUC |
|--------|-----------|-------------|--------------|
| Logistic Regression | 0.2573 | 0.65 | 0.6493 |
| Random Forest | 0.0183 | 0.03 | 0.5128 |
| Gradient Boosting | 0.0501 | 0.07 | 0.5315 |
| XGBoost | 0.0093 | 0.02 | 0.5100 |

**Modelo seleccionado:** Logistic Regression (class_weight=balanced, SMOTE)  
**Criterio de selección:** Selection Score = Performance (60% F-beta) + Consistency (25% estabilidad CV) + Scalability (15% velocidad de entrenamiento)

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

## Autor

**Alexandra Vasco Lopera**  
- GitHub: [@botanicalex](https://github.com/botanicalex)
- Email: alexavascolopera@gmail.com

## Licencia

Este proyecto es parte de un entregable académico — Ciencia de Datos en Producción 2026-1.