import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
from LinealRe import predecir_nota  # Importamos la función de regresión lineal
import LogisticRe  # Importamos la regresión logística

app = Flask(__name__)

#  Configuración de la conexión a MySQL
app.config['MYSQL_HOST'] = 'localhost'  # Cambia esto si usas un servidor externo
app.config['MYSQL_USER'] = 'root'  # Usuario de MySQL (por defecto "root")
app.config['MYSQL_PASSWORD'] = ''  # Si tienes contraseña en MySQL, agrégala aquí
app.config['MYSQL_DB'] = 'machinelearningdb'  # Nombre de la base de datos
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # Retorna los resultados como diccionarios

mysql = MySQL(app)

@app.route("/")
def home():
    return render_template("index.html")

#  Ruta que muestra los modelos de Machine Learning almacenados en la BD
@app.route("/modelos")
def mostrar_modelos():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM modelosml;")
        modelos = cur.fetchall()
        cur.close()
        return render_template("modelos.html", modelos=modelos)
    except Exception as e:
        return f"Error al obtener los modelos: {e}"

#  Ruta para agregar un nuevo modelo de Machine Learning
@app.route("/agregar_modelo", methods=["POST"])
def agregar_modelo():
    if request.method == "POST":
        nombre_modelo = request.form["nombre_modelo"]
        id_tipo_ml = request.form["id_tipo_ml"]

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO modelosml (nombre_modelo, id_tipo_ml) VALUES (%s, %s)", (nombre_modelo, id_tipo_ml))
        mysql.connection.commit()
        cur.close()
        return "Modelo agregado correctamente"

#  Ruta para predecir notas con regresión lineal
@app.route("/CalculoNota", methods=["GET", "POST"])
def calcular_nota():
    nota_predicha = None
    if request.method == "POST":
        try:
            horas = float(request.form["horas"])  # Convertir entrada a float
            nota_predicha = predecir_nota(horas)  # Llamar a la función de predicción
        except ValueError:
            nota_predicha = "Entrada inválida"

    return render_template("CalculoNota.html", nota_predicha=nota_predicha)

#    Ruta para predecir el abandono de clientes con regresión logística
@app.route("/prediccion_abandono", methods=["GET", "POST"])
def prediccion_abandono():
    resultado = None
    if request.method == "POST":
        try:
            historial = float(request.form["historial"])
            frecuencia = float(request.form["frecuencia"])
            gasto = float(request.form["gasto"])

            resultado = LogisticRe.predecir_abandono(historial, frecuencia, gasto)
        except ValueError:
            resultado = "Entrada no válida"

    return render_template("abandono_cliente.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
