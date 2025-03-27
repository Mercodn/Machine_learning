import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Datos de entrenamiento (simulados)
data = {
    "Historial_Compras": [5, 20, 10, 50, 30, 5, 40, 25, 15, 8],
    "Frecuencia_Compra": [30, 5, 20, 2, 10, 40, 3, 7, 25, 35],
    "Gasto_Promedio": [100, 500, 300, 700, 400, 120, 650, 450, 350, 150],
    "Abandono": [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]  # 1 = Abandona, 0 = Permanece
}

df = pd.DataFrame(data)

# Variables de entrada (X) y salida (y)
X = df[["Historial_Compras", "Frecuencia_Compra", "Gasto_Promedio"]]
y = df["Abandono"]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo de Regresión Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Función para predecir si un cliente abandonará o no
def predecir_abandono(historial, frecuencia, gasto):
    entrada = np.array([[historial, frecuencia, gasto]])
    entrada_escalada = scaler.transform(entrada)  # Normalizar entrada
    prediccion = model.predict(entrada_escalada)[0]
    return "Posible Abandono" if prediccion == 1 else "Cliente Leal"
