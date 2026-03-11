# ─────────────────────────────────────────────
# Imagen base: Python 3.11 slim
# slim reduce el tamaño de la imagen eliminando
# herramientas de desarrollo innecesarias
# ─────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="Alexandra Vasco Lopera"
LABEL description="API de predicción de mora crediticia"
LABEL version="1.0.0"

# ─────────────────────────────────────────────
# Variables de entorno
# ─────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ─────────────────────────────────────────────
# Directorio de trabajo dentro del contenedor
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Instalar dependencias del sistema
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Copiar e instalar dependencias Python primero
# (capa separada para aprovechar cache de Docker)
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copiar código fuente
# ─────────────────────────────────────────────
COPY mlops_pipeline/src/ .

# ─────────────────────────────────────────────
# Exponer puerto de la API
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# Comando de arranque
# reload=False en producción (a diferencia del desarrollo)
# ─────────────────────────────────────────────
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]