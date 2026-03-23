from flask import Flask, render_template, request
import numpy as np
from linear_regression_model import model, le_sex, le_smoker, le_region

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ml_use_cases')
def ml_use_cases():
    return render_template('ml_use_cases.html')

@app.route('/use_case_1')
def use_case_1():
    return render_template('use_case_1.html')

@app.route('/use_case_2')
def use_case_2():
    return render_template('use_case_2.html')

@app.route('/use_case_3')
def use_case_3():
    return render_template('use_case_3.html')

@app.route('/use_case_4')
def use_case_4():
    return render_template('use_case_4.html')

@app.route('/supervised_ml')
def supervised_ml():
    return render_template('supervised_ml.html')

@app.route('/linear_regression_concepts')
def linear_regression_concepts():
    return render_template('linear_regression_concepts.html')

@app.route('/linear_regression_app')
def linear_regression_app():
    return render_template('linear_regression_app.html')

@app.route('/linear_regression_dataset')
def linear_regression_dataset():
    return render_template('linear_regression_dataset.html')

@app.route('/linear_regression_variables')
def linear_regression_variables():
    return render_template('linear_regression_variables.html')

@app.route('/linear_regression_prep')
def linear_regression_prep():
    return render_template('linear_regression_prep.html')

@app.route('/linear_regression_viz')
def linear_regression_viz():
    return render_template('linear_regression_viz.html')

@app.route('/linear_regression_predict', methods=['GET', 'POST'])
def linear_regression_predict():
    prediction = None
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = le_sex.transform([request.form['sex']])[0]
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = le_smoker.transform([request.form['smoker']])[0]
        region = le_region.transform([request.form['region']])[0]
        
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
    
    return render_template('linear_regression_predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)