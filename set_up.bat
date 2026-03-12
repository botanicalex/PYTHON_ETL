@echo off
echo ================================================
echo   PYTHON_ETL - Configuracion inicial
echo   Ciencia de Datos en Produccion 2026-1
echo ================================================
echo.

REM Verificar que Python este instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado. Instala Python 3.10 o superior.
    pause
    exit /b 1
)

echo [OK] Python encontrado:
python --version
echo.

REM Crear entorno virtual
echo [1/3] Creando entorno virtual...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado en .venv
echo.

REM Activar entorno virtual
echo [2/3] Activando entorno virtual...
call .venv\Scripts\activate.bat
echo [OK] Entorno virtual activado
echo.

REM Instalar dependencias
echo [3/3] Instalando dependencias desde requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] No se pudieron instalar las dependencias.
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas correctamente
echo.

echo ================================================
echo   Instalacion completada exitosamente
echo.
echo   Para activar el entorno virtual:
echo     .venv\Scripts\activate
echo.
echo   Para correr la API:
echo     cd mlops_pipeline\src
echo     python model_deploy.py
echo.
echo   API disponible en: http://localhost:8000
echo   Documentacion en: http://localhost:8000/docs
echo ================================================
pause