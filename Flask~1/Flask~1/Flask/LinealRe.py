import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


horas_estudio = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
notas = np.array([38, 40, 44, 45, 50])


modelo = LinearRegression()
modelo.fit(horas_estudio, notas)


joblib.dump(modelo, "modelo_regresion.pkl")

def predecir_nota(horas):
    """Carga el modelo y predice la nota según las horas de estudio."""
    modelo_cargado = joblib.load("modelo_regresion.pkl")
    return round(modelo_cargado.predict([[horas]])[0], 2)
