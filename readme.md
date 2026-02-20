# PYTHON_ETL - Análisis de Crédito Financiero

**Ciencia de Datos en Producción - Entregable 2 (15%)**  
**Docente:** Juan Sebastián Parra Sánchez

## Descripción del Proyecto

Este proyecto forma parte de la evaluación para el equipo de Datos y Analítica de una empresa financiera. El objetivo es analizar datos de crédito (información del crédito, usuario y pago) para generar insights valiosos y evaluar la capacidad de comunicación efectiva de hallazgos.

## Estructura del Proyecto
```
PYTHON_ETL/
├── codigoproyecto-venv/          # Entorno virtual (excluido de Git)
├── etl_scripts/
│   └── src/
│       ├── desarrollo/
│       │   ├── __pycache__/      # Archivos compilados de Python
│       │   └── transformacion_eda.ipynb  # Notebook principal de análisis
│       └── config.json           # Configuración del proyecto
├── Base_de_datos.csv             # Archivo fuente de datos
├── requirements.txt              # Dependencias del proyecto
├── .gitignore                    # Exclusiones para Git
├── readme.md                     # Documentación del proyecto
└── set_up.bat                    # Script de configuración inicial
```

## Instalación

### Prerrequisitos
- Python 3.8 o superior
- Git

### Pasos de instalación

1. Clona el repositorio:
```bash
git clone https://github.com/botanicalex/NOMBRE-DEL-REPO.git
cd NOMBRE-DEL-REPO
```

2. Ejecuta el script de configuración:
```bash
set_up.bat
```

Este script:
- Crea el entorno virtual
- Instala todas las dependencias
- Verifica la instalación

## Uso

### 1. Activar el entorno virtual
```bash
codigoproyecto-venv\Scripts\activate
```

### 2. Iniciar Jupyter Notebook
```bash
jupyter notebook
```

### 3. Abrir el notebook principal

Navega a: etl_scripts/src/desarrollo/transformacion_eda.ipynb

## Contenido del Análisis

El notebook incluye:

- **Carga de datos**: Importación del DataFrame de crédito
- **Limpieza de datos**: Manejo de valores nulos y duplicados
- **Análisis Exploratorio (EDA)**:
  - Estadísticas descriptivas
  - Distribuciones de variables
  - Análisis de mora y pagos
  - Correlaciones entre variables
- **Visualizaciones**: Gráficos para comunicar hallazgos
- **Insights**: Conclusiones y recomendaciones

## Tecnologías Utilizadas

- **Python 3.x**
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib/Seaborn**: Visualizaciones
- **Jupyter Notebook**: Entorno interactivo

## Autor

**Alexandra Vasco Lopera**  
- GitHub: [@botanicalex](https://github.com/botanicalex)
- Email: alexavascolopera@gmail.com

## Licencia

Este proyecto es parte de un entregable académico.
