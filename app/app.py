
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path
from datetime import date



# CONFIGURACIÓN DE LA PÁGINA


st.set_page_config(
    page_title="Pronóstico de Ocupación - Barcelona",
    layout="centered"
)



# CARGAR MODELO


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "models" / "xgboost_final.joblib"


@st.cache_resource
def cargar_modelo():
    pipeline = joblib.load(RUTA_MODELO)

    preprocessor = pipeline.named_steps["preprocessor"]
    modelo_xgb = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(modelo_xgb)

    return pipeline, preprocessor, explainer


modelo, preprocessor, explainer = cargar_modelo()



# TÍTULO


st.title("Pronóstico de Ocupación por Barrio")

st.write(
    "Esta aplicación estima la tasa de ocupación de alojamientos "
    "por barrio en Barcelona utilizando un modelo XGBoost."
)



# BARRIOS UTILIZADOS DURANTE EL ENTRENAMIENTO


barrios = [
    "Can Baró",
    "Diagonal Mar i el Front Marítim del Poblenou",
    "Horta",
    "Hostafrancs",
    "Navas",
    "Pedralbes",
    "Porta",
    "Provençals del Poblenou",
    "Sant Andreu",
    "Sant Antoni",
    "Sant Gervasi - Galvany",
    "Sant Gervasi - la Bonanova",
    "Sant Martí de Provençals",
    "Sant Pere, Santa Caterina i la Ribera",
    "Sants",
    "Sants - Badal",
    "Sarrià",
    "Vallcarca i els Penitents",
    "Vallvidrera, el Tibidabo i les Planes",
    "Vilapicina i la Torre Llobeta",
    "el Baix Guinardó",
    "el Barri Gòtic",
    "el Besòs i el Maresme",
    "el Camp d'en Grassot i Gràcia Nova",
    "el Camp de l'Arpa del Clot",
    "el Carmel",
    "el Clot",
    "el Coll",
    "el Congrés i els Indians",
    "el Fort Pienc",
    "el Guinardó",
    "el Parc i la Llacuna del Poblenou",
    "el Poble Sec",
    "el Poblenou",
    "el Putxet i el Farró",
    "el Raval",
    "el Turó de la Peira",
    "l'Antiga Esquerra de l'Eixample",
    "la Barceloneta",
    "la Bordeta",
    "la Dreta de l'Eixample",
    "la Font d'en Fargues",
    "la Font de la Guatlla",
    "la Guineueta",
    "la Marina de Port",
    "la Maternitat i Sant Ramon",
    "la Nova Esquerra de l'Eixample",
    "la Prosperitat",
    "la Sagrada Família",
    "la Sagrera",
    "la Salut",
    "la Teixonera",
    "la Verneda i la Pau",
    "la Vila Olímpica del Poblenou",
    "la Vila de Gràcia",
    "les Corts",
    "les Roquetes",
    "les Tres Torres"
]



# FORMULARIO


st.subheader("Datos para realizar el pronóstico")

barrio = st.selectbox(
    "Barrio",
    barrios
)

fecha = st.date_input(
    "Fecha a pronosticar",
    value=date.today()
)

lag_7_pct = st.slider(
    "Ocupación estimada hace 7 días (%)",
    min_value=0,
    max_value=100,
    value=50
)

lag_14_pct = st.slider(
    "Ocupación estimada hace 14 días (%)",
    min_value=0,
    max_value=100,
    value=50
)

lag_28_pct = st.slider(
    "Ocupación estimada hace 28 días (%)",
    min_value=0,
    max_value=100,
    value=50
)


# PREDICCIÓN
if st.button("Realizar pronóstico"):

    # Convertir porcentajes a escala 0-1
    lag_7 = lag_7_pct / 100
    lag_14 = lag_14_pct / 100
    lag_28 = lag_28_pct / 100

    # Variables temporales calculadas automáticamente
    day_of_week = fecha.weekday()
    month = fecha.month

    # Crear observación
    datos = pd.DataFrame([{
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "day_of_week": day_of_week,
        "month": month,
        "neighbourhood_cleansed": barrio
    }])


    # PREDICCIÓN XGBOOST
    prediccion = modelo.predict(datos)[0]
    prediccion = max(0, min(1, prediccion))

    st.success(
        f"Ocupación estimada: {prediccion * 100:.2f}%"
    )

    # Rango estimado basado en validación temporal

    ERROR_P90 = 0.050693

    limite_inferior = max(0.0, prediccion - ERROR_P90)
    limite_superior = min(1.0, prediccion + ERROR_P90)

    st.info(
        f"Rango estimado (90%): "
        f"{limite_inferior * 100:.2f}% – "
        f"{limite_superior * 100:.2f}%"
    )

    st.caption(
        "La estimación se basa en la disponibilidad publicada de alojamientos "
        "y no representa reservas confirmadas."
    )

    st.caption(
        "El rango se calcula a partir del percentil 90 de los errores absolutos "
        "observados durante la validación temporal del modelo."
    )



    # EXPLICACIÓN SHAP

    st.subheader("¿Qué influyó en este pronóstico?")

    # Aplicar exactamente el mismo preprocesamiento
    datos_proc = preprocessor.transform(datos)

    # Obtener nombres después del preprocesamiento
    nombres_features = preprocessor.get_feature_names_out()

    # Calcular valores SHAP
    shap_values = explainer.shap_values(datos_proc)

    # Tabla completa de contribuciones
    explicacion = pd.DataFrame({
        "variable": nombres_features,
        "shap_value": shap_values[0]
    })


    # AGRUPAR LAS 58 VARIABLES DE BARRIO
    contribuciones = {
        "Ocupación hace 7 días":
            explicacion.loc[
                explicacion["variable"] == "num__lag_7",
                "shap_value"
            ].sum(),

        "Ocupación hace 14 días":
            explicacion.loc[
                explicacion["variable"] == "num__lag_14",
                "shap_value"
            ].sum(),

        "Ocupación hace 28 días":
            explicacion.loc[
                explicacion["variable"] == "num__lag_28",
                "shap_value"
            ].sum(),

        "Día de la semana":
            explicacion.loc[
                explicacion["variable"] == "num__day_of_week",
                "shap_value"
            ].sum(),

        "Mes":
            explicacion.loc[
                explicacion["variable"] == "num__month",
                "shap_value"
            ].sum(),

        "Barrio":
            explicacion.loc[
                explicacion["variable"].str.startswith(
                    "cat__neighbourhood_cleansed_"
                ),
                "shap_value"
            ].sum()
    }


    # CREAR TABLA AMIGABLE

    df_contribuciones = pd.DataFrame({
        "Factor": contribuciones.keys(),
        "Contribución": contribuciones.values()
    })

    df_contribuciones["Importancia"] = (
        df_contribuciones["Contribución"].abs()
    )

    df_contribuciones = df_contribuciones.sort_values(
        "Importancia",
        ascending=False
    )

    # EXPLICACIÓN EN LENGUAJE NATURAL

    factor_principal = df_contribuciones.iloc[0]

    direccion = (
        "hacia arriba"
        if factor_principal["Contribución"] > 0
        else "hacia abajo"
    )

    st.write(
        f"El factor que más influyó en esta predicción fue "
        f"**{factor_principal['Factor']}**, empujando la "
        f"estimación **{direccion}**."
    )


      # --------------------------------------------------------
    # GRÁFICA SHAP PERSONALIZADA
    # --------------------------------------------------------

    import matplotlib.pyplot as plt
    import textwrap

    df_grafico = df_contribuciones.copy()

    # Etiquetas en máximo 2 líneas
    def dividir_etiqueta(texto):
        palabras = texto.split()

        if len(texto) <= 18:
            return texto

        mitad = len(palabras) // 2

        return (
            " ".join(palabras[:mitad])
            + "\n"
            + " ".join(palabras[mitad:])
        )

    etiquetas = [
        dividir_etiqueta(x)
        for x in df_grafico["Factor"]
    ]

    valores = df_grafico["Contribución"].values

    # Crear gráfica
    fig, ax = plt.subplots(figsize=(9, 5))

    barras = ax.bar(
        etiquetas,
        valores
    )

    # Línea de referencia en cero
    ax.axhline(0, linewidth=1)

    # Valores en las barras
    for barra, valor in zip(barras, valores):

        if abs(valor) > 0.008:

            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor / 2,
                f"{valor:+.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold"
            )

        else:

            desplazamiento = 0.002 if valor >= 0 else -0.002

            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + desplazamiento,
                f"{valor:+.3f}",
                ha="center",
                va="bottom" if valor >= 0 else "top",
                fontsize=9
            )

    ax.set_ylabel("Contribución SHAP")
    ax.set_xlabel("")
    ax.set_title(
        "Influencia de cada factor en la predicción",
        fontsize=13
    )

    ax.tick_params(
        axis="x",
        labelrotation=0
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    st.caption(
            "Los valores positivos aumentan la estimación respecto "
            "al valor base del modelo y los valores negativos la reducen. "
        )