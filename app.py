from flask import Flask, render_template, request
import numpy as np
from linear_regression_model import model, le_sex, le_smoker, le_region
from logistic_regression_model import model as logistic_model
from perceptron_model import model as perceptron_model

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

@app.route('/logistic_regression_concepts')
def logistic_regression_concepts():
    return render_template('logistic_regression_concepts.html')

@app.route('/logistic_regression_app')
def logistic_regression_app():
    return render_template('logistic_regression_app.html')

@app.route('/perceptron_concepts')
def perceptron_concepts():
    return render_template('perceptron_concepts.html')

@app.route('/perceptron_app')
def perceptron_app():
    return render_template('perceptron_app.html')

@app.route('/logistic_regression_dataset')
def logistic_regression_dataset():
    return render_template('logistic_regression_dataset.html')

@app.route('/logistic_regression_variables')
def logistic_regression_variables():
    return render_template('logistic_regression_variables.html')

@app.route('/logistic_regression_prep')
def logistic_regression_prep():
    return render_template('logistic_regression_prep.html')

@app.route('/logistic_regression_viz')
def logistic_regression_viz():
    return render_template('logistic_regression_viz.html')

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

@app.route('/logistic_regression_predict', methods=['GET', 'POST'])
def logistic_regression_predict():
    prediction = None
    if request.method == 'POST':
        length = float(request.form['length'])
        num_exclamation = int(request.form['num_exclamation'])
        num_dollar = int(request.form['num_dollar'])
        num_uppercase = int(request.form['num_uppercase'])
        
        input_data = np.array([[length, num_exclamation, num_dollar, num_uppercase]])
        prediction = logistic_model.predict(input_data)[0]
        prediction = 'Spam' if prediction == 1 else 'Not Spam'
    
    return render_template('logistic_regression_predict.html', prediction=prediction)

@app.route('/perceptron_dataset')
def perceptron_dataset():
    return render_template('perceptron_dataset.html')

@app.route('/perceptron_variables')
def perceptron_variables():
    return render_template('perceptron_variables.html')

@app.route('/perceptron_prep')
def perceptron_prep():
    return render_template('perceptron_prep.html')

@app.route('/perceptron_viz')
def perceptron_viz():
    return render_template('perceptron_viz.html')

@app.route('/perceptron_predict', methods=['GET', 'POST'])
def perceptron_predict():
    prediction = None
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = perceptron_model.predict(input_data)[0]
        prediction = 'Versicolor' if prediction == 1 else 'Setosa'
    
    return render_template('perceptron_predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)