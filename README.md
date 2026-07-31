# ¿Cuánto vale un departamento en Buenos Aires?

## Predicción de precios inmobiliarios con aprendizaje automático

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Regression-2E7D32)
![Real Estate](https://img.shields.io/badge/Real%20Estate-CABA-455A64)

Se desarrolla un proceso de predicción del precio de propiedades de **1 a 3 ambientes** en la Ciudad de Buenos Aires. El análisis parte de la base completa de Properati, documenta la preparación de los avisos mediante 18 etapas y compara Random Forest, KNeighbors e HistGradientBoosting.

## Descripción del proyecto

La base original contiene **1.000.000 de avisos** y 25 columnas en el archivo, de las cuales 24 se analizan luego de utilizar `id` como índice. El estudio filtra Capital Federal, corrige la ubicación, trata datos faltantes, convierte precios a dólares, selecciona tipologías y operaciones, imputa características por barrio y analiza valores atípicos.

Puerto Madero se separa del resto de CABA debido a su escala de precios y comportamiento particular. El notebook conserva la secuencia completa del análisis y los resultados obtenidos en cada etapa.

## Objetivo

Construir un modelo de predicción del precio publicado de departamentos, casas y PH de uno a tres ambientes en Capital Federal, evaluando cómo la preparación de datos y la elección del algoritmo modifican el error de estimación.

## Datos incluidos

La carpeta `data/` contiene la base original y dos checkpoints limpios:

- **Base original:** `properati.csv.zip`, dividida en 12 partes para respetar el límite de carga del navegador de GitHub.
- **Puerto Madero:** `df3_limpio.csv`, con 1,035 observaciones y 19 variables.
- **Resto de CABA:** `df4_limpio.csv.gz`, con 59,991 observaciones y 19 variables.

Las partes no deben descomprimirse, eliminarse ni renombrarse. El notebook las une automáticamente y abre `properati.csv` desde el ZIP reconstruido.

```text
data/
│   ├── properati.csv.zip.part01
│   ├── properati.csv.zip.part02
│   ├── properati.csv.zip.part03
│   ├── properati.csv.zip.part04
│   ├── properati.csv.zip.part05
│   ├── properati.csv.zip.part06
│   ├── properati.csv.zip.part07
│   ├── properati.csv.zip.part08
│   ├── properati.csv.zip.part09
│   ├── properati.csv.zip.part10
│   ├── properati.csv.zip.part11
│   ├── properati.csv.zip.part12
│   ├── df3_limpio.csv
│   └── df4_limpio.csv.gz
```

El archivo reconstruido `properati.csv.zip` está ignorado por Git porque supera el límite de tamaño de un archivo individual. Solo se versionan sus partes.

## Preparación de los datos

El proceso desarrollado comprende:

1. Exploración de la base original y filtrado de Capital Federal.
2. Corrección del nombre de las columnas de latitud y longitud.
3. Eliminación de variables irrelevantes o con alta proporción de faltantes.
4. Imputación y depuración de barrio (`l3`).
5. Imputación de coordenadas y control de límites geográficos de CABA.
6. Conversión de precios en pesos a dólares y eliminación de publicaciones sin precio.
7. Selección de casas, departamentos y PH.
8. Selección del tipo de operación estudiado.
9. Imputación de ambientes, dormitorios, baños y superficies mediante estadísticas por barrio.
10. Separación de Puerto Madero y tratamiento específico de valores atípicos.
11. Construcción de precios de referencia por metro cuadrado y barrio.
12. Comparación de modelos predictivos con búsqueda de hiperparámetros.

## Evolución del RMSE

Los árboles de decisión de profundidad tres se utilizaron como referencia durante la preparación. Algunos resultados destacados fueron:

| Etapa | Cambio acumulado | RMSE registrado |
|---:|---|---:|
| 1 | Línea de base | 899.118 |
| 6 | Selección del tipo de operación | 87.117 |
| 11 | Imputación de superficie cubierta | 162.480 |
| 15 | Tratamiento de outliers en el resto de CABA | 17.204 |
| 16 | Línea de base de Puerto Madero | 228.366 |
| 17 | Tratamiento específico de Puerto Madero | 71.969 |

La evolución completa se encuentra en `results/iterative_model_results.csv`.

## Modelos predictivos y resultados registrados

| Segmento | Modelo | RMSE promedio | Validación | Mejores parámetros |
|---|---|---:|---:|---|
| Resto de CABA | Random Forest | 12.184 | 7 folds | 300 árboles, profundidad 15, 5 variables por división |
| Resto de CABA | KNeighbors | 12.237 | 10 folds | 30 vecinos, distancia, métrica cityblock |
| Resto de CABA | HistGradientBoosting | 12.727 | 10 folds | 1.000 iteraciones, profundidad 18, learning rate 0,4, L2 0,1 |
| Puerto Madero | Random Forest | 23.228 | 7 folds | 300 árboles, profundidad 15, 7 variables por división |

Random Forest obtuvo el menor RMSE registrado en el resto de CABA y fue el modelo aplicado también al segmento de Puerto Madero.

## Consideración metodológica

En las etapas finales se construyen `precio_ajustado` y `diferencia_precio_real_ajustado` a partir del precio publicado y del precio por metro cuadrado de cada barrio. Estas variables resultan útiles para el análisis de outliers y para explicar los resultados obtenidos, pero contienen información derivada de la variable objetivo. Por ese motivo, las métricas deben interpretarse como resultados exploratorios del proceso desarrollado y no como una evaluación definitiva de un modelo listo para estimar propiedades nuevas.

Esta observación se incorpora sin reemplazar ni modificar los resultados registrados en el notebook.

## Estructura del repositorio

```text
caba-apartment-price-prediction/
├── assets/figures/      # Gráficos basados en los resultados registrados
├── data/                # Base original dividida y checkpoints limpios
├── notebooks/           # Notebook completo con rutas relativas
├── reports/             # Informe profesional en PDF y Word
├── results/             # Resultados, inventario y manifiesto de archivos
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/caba-apartment-price-prediction.ipynb
```

La primera ejecución reconstruye `data/properati.csv.zip` a partir de las partes. El archivo reconstruido puede ocupar aproximadamente 237 MB y no debe agregarse al repositorio.

## Principales conclusiones

- La preparación de datos tuvo un efecto decisivo sobre el RMSE registrado.
- La conversión de moneda, la selección de operaciones y el tratamiento de superficies fueron etapas centrales.
- Puerto Madero requirió un análisis separado por su escala de precios y menor cantidad de observaciones.
- Random Forest obtuvo el mejor RMSE promedio registrado para el resto de CABA.
- La superficie cubierta fue la variable de mayor importancia registrada en Puerto Madero.
- Los indicadores construidos con el precio deben reservarse para exploración y control de outliers.

## Informe

El informe completo se encuentra en `reports/` y desarrolla el objetivo, los datos, la preparación, las 18 etapas, los modelos, los resultados, las limitaciones, la bibliografía y el anexo.

## Autora

**Lea Lambrecht**  
[LinkedIn](https://www.linkedin.com/in/lealambrecht/) · [GitHub](https://github.com/leanlambrecht)
