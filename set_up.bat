@echo off
echo ================================================
echo   CONFIGURACION DEL PROYECTO PYTHON_ETL
echo   Ciencia de Datos en Produccion - Entregable 2
echo ================================================
echo.

echo [1/4] Verificando Python...
python --version
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    pause
    exit /b 1
)
echo.

echo [2/4] Creando entorno virtual...
if exist codigoproyecto-venv (
    echo El entorno virtual ya existe. Eliminando...
    rmdir /s /q codigoproyecto-venv
)
python -m venv codigoproyecto-venv
echo.

echo [3/4] Activando entorno virtual e instalando dependencias...
call codigoproyecto-venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

echo [4/4] Verificando instalacion...
pip list
echo.

echo ================================================
echo   CONFIGURACION COMPLETADA EXITOSAMENTE
echo ================================================
echo.
echo Para activar el entorno virtual en el futuro, ejecuta:
echo    codigoproyecto-venv\Scripts\activate.bat
echo.
echo Para abrir Jupyter Notebook:
echo    jupyter notebook
echo.
pause
