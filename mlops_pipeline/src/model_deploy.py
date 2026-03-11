import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

# ─────────────────────────────────────────────
# 1. Esquema de entrada
# Columnas que produce ft_engineering (X_train.columns)
# tipo_credito y tipo_laboral como str (no object)
# ─────────────────────────────────────────────

class PredictionInput(BaseModel):
    tipo_credito              : str
    capital_prestado          : float
    plazo_meses               : float
    edad_cliente              : float
    tipo_laboral              : str
    salario_cliente           : float
    total_otros_prestamos     : float
    puntaje_datacredito       : float
    huella_consulta           : float
    saldo_total               : float
    creditos_sectorFinanciero : float
    creditos_sectorCooperativo: float
    creditos_sectorReal       : float
    tiene_mora                : int
    ratio_capital_salario     : float


class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]


# ─────────────────────────────────────────────
# 2. Inicialización de la app y carga del modelo
# ─────────────────────────────────────────────

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="Despliega el mejor modelo entrenado para predicciones por batch.",
    version="1.0.0"
)

# Imports de módulos internos al inicio — evita imports dentro de funciones
try:
    from model_evaluation import evaluation
    _evaluation_available = True
except ImportError:
    _evaluation_available = False

try:
    from model_monitoring import detectar_data_drift
    _monitoring_available = True
except ImportError:
    _monitoring_available = False

# Carga del modelo al iniciar la app
try:
    model = joblib.load("mejor_modelo.pkl")
    print("Modelo cargado correctamente.")
    _model_loaded = True
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None
    _model_loaded = False


# ─────────────────────────────────────────────
# 3. Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "status" : "OK" if _model_loaded else "ERROR",
        "modelo" : "cargado" if _model_loaded else "no disponible",
        "message": "API de predicción en línea. Usa el endpoint /predict para predicciones."
    }


@app.get("/health")
def health_check():
    """
    Health check del modelo.
    Retorna 503 si el modelo no está disponible — útil para
    orquestadores como Kubernetes o load balancers.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica que mejor_modelo.pkl existe."
        )
    return {"status": "healthy", "modelo": "disponible"}


@app.post("/predict")
async def predict_batch(input_data: BatchPredictionInput):
    """
    Endpoint para predicciones por batch.
    Recibe una lista de registros y devuelve predicciones con
    probabilidad de mora por cada registro.
    El pipeline incluye preprocesamiento completo internamente.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible."
        )

    try:
        input_list = [item.dict() for item in input_data.data]
        df = pd.DataFrame(input_list)

        predictions  = model.predict(df).tolist()
        probabilities = model.predict_proba(df)[:, 0].tolist()  # prob mora (clase 0)

        resultados = [
            {
                "prediccion"       : int(pred),
                "etiqueta"         : "Paga a tiempo" if pred == 1 else "Mora",
                "probabilidad_mora": round(prob, 4),
            }
            for pred, prob in zip(predictions, probabilities)
        ]

        return {"predictions": resultados}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error durante la predicción: {str(e)}"
        )


@app.get(
    "/evaluation",
    responses={200: {"content": {"image/png": {}}}},
    description="Retorna imagen PNG con métricas de evaluación del modelo desplegado."
)
async def serve_evaluation_plot():
    """
    Endpoint para visualizar la evaluación del modelo.
    Retorna un dashboard PNG con métricas, matriz de confusión,
    curva ROC y curva Precision-Recall.
    """
    if not _evaluation_available:
        raise HTTPException(
            status_code=501,
            detail="model_evaluation.py no encontrado."
        )
    try:
        image_buffer = evaluation()
        if image_buffer:
            return Response(content=image_buffer.getvalue(), media_type="image/png")
        raise HTTPException(
            status_code=500,
            detail="No se pudo generar el gráfico de evaluación."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando evaluación: {str(e)}"
        )


@app.post("/monitor", tags=["Monitoreo"])
async def monitor_data_drift(datos_produccion: BatchPredictionInput):
    """
    Recibe un lote de datos de producción y lo compara con el
    dataset de referencia para detectar Data Drift usando el test KS.
    """
    if not _monitoring_available:
        raise HTTPException(
            status_code=501,
            detail="model_monitoring.py no encontrado."
        )
    try:
        df_produccion = pd.DataFrame([item.dict() for item in datos_produccion.data])
        reporte = detectar_data_drift(df_produccion)

        if reporte.empty:
            return {"status": "No se encontraron columnas numéricas comunes para analizar."}

        reporte_json = reporte.to_dict("records")
        hay_drift    = any(item["drift_detectado"] for item in reporte_json)

        return {
            "status"       : "Drift detectado" if hay_drift else "Sin drift significativo",
            "hay_drift"    : hay_drift,
            "reporte_drift": reporte_json,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en monitoreo: {str(e)}"
        )


# ─────────────────────────────────────────────
# 4. Ejecución
# ─────────────────────────────────────────────

# Ejecución local:
#   python model_deploy.py
#
# Ejecución con Docker (imagen con librerías y código):
#   docker build -t modelo-mora .
#   docker run -p 8000:8000 modelo-mora
#
# API disponible en : http://localhost:8000
# Documentación en  : http://localhost:8000/docs

if __name__ == "__main__":
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)