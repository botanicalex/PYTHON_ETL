from __future__ import annotations

import joblib
import pandas as pd
import uvicorn

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# 1. Esquema de entrada / salida
# ─────────────────────────────────────────────

class PredictionInput(BaseModel):
    tipo_credito              : Any
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
    data: List[PredictionInput] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    prediccion        : int
    etiqueta          : str
    probabilidad_mora : float


class BatchPredictionResponse(BaseModel):
    n_records  : int
    predictions: List[PredictionResult]


# ─────────────────────────────────────────────
# 2. Servicio de modelo
# ─────────────────────────────────────────────

class ModelDeploymentService:
    """
    Encapsula la carga y uso del modelo serializado.
    Separa la lógica del modelo de la capa HTTP (FastAPI).
    """
    def __init__(self, model_path: Path = Path("mejor_modelo.pkl")):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.model_path}. "
                "Ejecuta model_training.py primero."
            )
        model = joblib.load(self.model_path)
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "El modelo cargado no implementa 'predict_proba'. "
                "Asegurate de desplegar un modelo que soporte probabilidades."
            )
        return model

    def predict_batch(self, records: List[Dict[str, Any]]) -> BatchPredictionResponse:
        df = pd.DataFrame(records)
        predictions  = self.model.predict(df).tolist()
        probabilities = self.model.predict_proba(df)[:, 0].tolist()  # prob mora (clase 0)

        resultados = [
            PredictionResult(
                prediccion        = int(pred),
                etiqueta          = "Paga a tiempo" if pred == 1 else "Mora",
                probabilidad_mora = round(prob, 4),
            )
            for pred, prob in zip(predictions, probabilities)
        ]
        return BatchPredictionResponse(n_records=len(records), predictions=resultados)


# ─────────────────────────────────────────────
# 3. Fábrica de la app
# ─────────────────────────────────────────────

def create_app(model_path: Path = Path("mejor_modelo.pkl")) -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    Si el modelo no se puede cargar, la API arranca en modo degradado
    y responde con status 503 en /predict.
    """
    app = FastAPI(
        title       = "API de Predicción de Pago a Tiempo",
        description = "Despliega el mejor modelo entrenado para predicciones por batch.",
        version     = "1.0.0",
    )

    # Intentar cargar el modelo al arrancar
    try:
        service       = ModelDeploymentService(model_path=model_path)
        startup_error = None
        print("Modelo cargado correctamente.")
    except Exception as e:
        service       = None
        startup_error = str(e)
        print(f" Error al cargar el modelo: {e}")

    # ── Endpoints ─────────────────────────────

    @app.get("/")
    def read_root() -> Dict[str, str]:
        return {
            "status" : "ok" if service is not None else "error",
            "message": "API de predicción en línea. Usa /predict para predicciones.",
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status"      : "ok" if service is not None else "error",
            "model_path"  : str(model_path),
            "model_loaded": service is not None,
            "detail"      : startup_error,
        }

    @app.post("/predict", response_model=BatchPredictionResponse)
    async def predict_batch(input_data: BatchPredictionInput) -> BatchPredictionResponse:
        """
        Predicciones por batch. Recibe una lista de registros preprocesados
        y devuelve predicción, etiqueta y probabilidad de mora por registro.
        """
        if service is None:
            raise HTTPException(status_code=503, detail=startup_error)
        try:
            records = [item.dict() for item in input_data.data]
            return service.predict_batch(records)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error durante la predicción: {e}") from e

    @app.get(
        "/evaluation",
        responses={200: {"content": {"image/png": {}}}},
        description="Retorna un dashboard PNG con las métricas del modelo desplegado.",
    )
    async def serve_evaluation_plot() -> Response:
        """Genera y retorna el dashboard de evaluación como imagen PNG."""
        try:
            from model_evaluation import evaluation
            image_buffer = evaluation()
            if image_buffer:
                return Response(content=image_buffer.getvalue(), media_type="image/png")
            raise HTTPException(status_code=500, detail="No se pudo generar el gráfico.")
        except ImportError as e:
            raise HTTPException(status_code=501, detail="model_evaluation.py no encontrado.") from e

    @app.post("/monitor", tags=["Monitoreo"])
    async def monitor_data_drift(datos_produccion: BatchPredictionInput) -> Dict[str, Any]:
        """
        Recibe un lote de datos de producción y detecta Data Drift
        comparando con el dataset de referencia (test KS).
        """
        try:
            from model_monitoring import detectar_data_drift
        except ImportError as e:
            raise HTTPException(status_code=501, detail="model_monitoring.py no encontrado.") from e

        try:
            df_produccion = pd.DataFrame([item.dict() for item in datos_produccion.data])

            if len(df_produccion) < 2:
                return {"status": "Muestra insuficiente para análisis de drift (mínimo 2 registros)."}

            reporte = detectar_data_drift(df_produccion)

            if reporte.empty:
                return {"status": "No se encontraron columnas numéricas en común. Verifique el esquema de los datos enviados."}

            reporte_json = reporte.to_dict("records")
            hay_drift    = any(item["drift_detectado"] for item in reporte_json)

            return {
                "status"       : "Drift detectado" if hay_drift else "Sin drift significativo",
                "hay_drift"    : hay_drift,
                "reporte_drift": reporte_json,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error en monitoreo: {e}") from e

    return app



# ─────────────────────────────────────────────
# 4. Instancia de la app y ejecución local
# ─────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    print("\nPara correr con Docker (desde la raiz del proyecto):")
    print("  docker build -t modelo-mora .")
    print("  docker run -p 8000:8000 modelo-mora")
    print("\nAPI disponible en: http://localhost:8000")
    print("Documentacion   : http://localhost:8000/docs\n")
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)