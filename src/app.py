
import pandas as pd
import os
import numpy as np
import joblib
import streamlit as st

# --- PASO 1: LA RUTA PARA NO PERDERNOS ---
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_al_modelo = os.path.join(directorio_actual, 'modelo_diabetes.joblib')

@st.cache_resource
def load_model():
    return joblib.load(ruta_al_modelo)

model = load_model()

# --- PASO 2: LOS WIDGETS (LAS PALANCAS) ---
st.title("🏥 Mi Doctor Virtual (El Beta)")
st.write("Epa, dale a estas palancas para ver qué lo qué con tu salud:")

# Creamos las barras para los datos principales
# Ponemos valores por defecto para que no se vea vacío
preg = st.slider("Embarazos", 0, 20, 1)
glu = st.slider("Glucosa", 0, 200, 100)
bp = st.slider("Presión Arterial", 0, 150, 70)
skin = st.slider("Grosor de Piel", 0, 100, 20)
ins = st.slider("Insulina", 0, 900, 80)
bmi = st.slider("IMC (Peso)", 0.0, 70.0, 30.0)
pedi = st.slider("Pedigrí Familiar", 0.0, 2.5, 0.5)
age = st.slider("Edad", 0, 100, 30)

# --- PASO 3: EL MOMENTO DE LA VERDAD ---
if st.button("¿Tengo el azúcar alta, mi llave?"):
    
    # 1. Calculamos los logaritmos que el modelo también pide
    # El modelo es exigente y quiere los datos normales Y los logarítmicos
    l_dia = np.log(glu + 1)
    l_pre = np.log(preg + 1)
    l_age = np.log(age + 1)
    l_bmi = np.log(bmi + 1)
    
    # 2. Armamos el DataFrame con las 12 columnas EXACTAS y en el ORDEN de la tabla
    # No puede faltar ni una porque si no, el modelo se achicopala
    datos = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, pedi, age, l_dia, l_pre, l_age, l_bmi]], 
                         columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 
                                  'log_Diabetes', 'log_Pregnacies', 'log_Age', 'log_BMI'])
    
    # 3. ¡Soltamos la predicción!
    prediccion = model.predict(datos)
    
    if prediccion[0] == 1:
        st.error("⚠️ Cuídate, mano, el modelo dice que tienes riesgo de diabetes.")
    else:
        st.success("✅ ¡Estás fino! El modelo dice que estás sano como un rábano.")

        