from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/LinearRegresion/", methods=["GET", "POST"])
def calculateGrade():
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)

    return render_template("LinearR.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)