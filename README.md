# Pronóstico de Ocupación por Barrio en Barcelona

Proyecto final de Machine Learning 2.

El proyecto utiliza datos de Inside Airbnb para pronosticar la tasa de ocupación estimada de alojamientos por barrio en Barcelona.

## Aplicación

La aplicación fue desarrollada con Streamlit y utiliza el modelo XGBoost entrenado para generar el pronóstico.

El usuario selecciona:
- Barrio
- Fecha a pronosticar
- Ocupación estimada 7 días antes
- Ocupación estimada 14 días antes
- Ocupación estimada 28 días antes

La aplicación genera una estimación de ocupación y utiliza SHAP para explicar la influencia de las variables en cada predicción.

## Ejecutar localmente

pip install -r requirements.txt

streamlit run app/app.py

## Nota

La ocupación utilizada en este proyecto es una estimación basada en la disponibilidad publicada de los alojamientos y no representa reservas confirmadas.