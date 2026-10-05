# Pronóstico de Ocupación por Barrio en Barcelona

Proyecto final de Machine Learning II: problema 7.

## Equipo

- José Atencio
- Karina Bosquez
- Joshua Campos

Profesor: Juan Montenegro.

## Objetivo y datos

Estimar la tasa de ocupación de alojamientos por barrio en Barcelona utilizando datos de Inside Airbnb. La variable objetivo, `occupancy_rate`, es la proporción de alojamientos publicados como no disponibles para un barrio y una fecha.

Los archivos de origen son `listings.csv.gz` y `calendar.csv.gz`. Para el modelado se utilizan 58 barrios con al menos 10 alojamientos y un período de cobertura completa entre el 03/07/2026 y el 23/06/2027. Se construyen rezagos de 7, 14 y 28 días por barrio.

## Evaluación y resultados

Se utilizan tres folds de validación temporal y se reservan los últimos 28 días, del 27/05/2027 al 23/06/2027, para el test final. La métrica principal es el error absoluto medio (MAE); un valor menor indica un mejor desempeño.

| Enfoque | MAE test |
|---|---:|
| Baseline Lag-7 | 0.025843 |
| XGBoost optimizado | 0.027244 |
| MLP optimizado | 0.033225 |
| Baseline media | 0.131012 |

El baseline Lag-7 obtiene el menor error en test. XGBoost es el modelo aprendido con mejor desempeño y se seleccionó para la aplicación. Su MAE equivale aproximadamente a 2.72 puntos porcentuales de error promedio en la tasa estimada.

## Aplicación

La aplicación está desarrollada con Streamlit y utiliza XGBoost para generar el pronóstico y SHAP para explicar la influencia de las variables en cada predicción.

**Aplicación publicada:** https://barcelona-occupancy-ml2.streamlit.app

El usuario selecciona un barrio y una fecha e introduce las tasas de ocupación estimada del mismo barrio correspondientes exactamente a 7, 14 y 28 días antes de la fecha seleccionada. En la interfaz estos valores se ingresan como porcentajes.

## Archivos principales

| Ruta | Contenido |
|---|---|
| `app/app.py` | Aplicación Streamlit |
| `requirements.txt` | Dependencias para ejecutar la aplicación |
| `notebooks/` | Notebook de preparación, entrenamiento y evaluación |
| `data/raw/` | Datos de origen |
| `data/processed/` | Datos preparados |
| `models/xgboost_final.joblib` | Modelo XGBoost final |
| `models/mlp_final.keras` | Red neuronal MLP final |
| `models/preprocessor_mlp_final.joblib` | Preprocesador del MLP |
| `models/baseline_media.joblib` | Baseline de media |
| `figures/` | Gráficos del proyecto |

## Ejecutar la aplicación localmente

Abre una terminal en la carpeta raíz del proyecto y ejecuta:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Abre la dirección local que muestra Streamlit. Para utilizar el app con los modelos finales ya guardados no es necesario volver a entrenarlos.

## Ejecutar el notebook

1. Coloca `listings.csv.gz` y `calendar.csv.gz` en `data/raw/`.
2. Abre el notebook definitivo de `notebooks/` en Jupyter con un entorno que tenga sus dependencias instaladas, incluidas TensorFlow/Keras y XGBoost.
3. Usa `notebooks/` como directorio de trabajo: el notebook define la raíz del proyecto mediante `Path("..")`.
4. Reinicia el kernel y ejecuta las celdas en orden, desde la configuración hasta la evaluación y exportación de modelos.
5. Comprueba que no aparezcan errores y que las métricas coincidan con los resultados documentados.

Las celdas de guardado pueden reemplazar los modelos en `models/`. Si vuelves a entrenarlos, revisa los resultados antes de actualizar el app publicado.

## Reproducibilidad

Se verificó que las predicciones comparadas de XGBoost, MLP, Baseline Lag-7 y Baseline media fueran iguales, con diferencia máxima de 0.0. Esta comprobación valida la consistencia de esas predicciones; no demuestra por sí sola que nuevos entrenamientos siempre produzcan resultados idénticos.

## Limitaciones

- La tasa representa no disponibilidad publicada, no reservas confirmadas. Un anfitrión también puede bloquear fechas por otros motivos.
- La evaluación es temporal con actualización de rezagos usando valores anteriores ya observados. El bloque de test abarca 28 días; esto no equivale a predecir los 28 días simultáneamente sin nueva información.
- Para una fecha futura, los rezagos deben estar disponibles. Si todavía no se conocen, sería necesario definir un procedimiento adicional para estimarlos.
- Los resultados reportados corresponden al período y a los barrios evaluados; el desempeño en otros períodos debe validarse.
