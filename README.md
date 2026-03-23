# Machine Learning Use Cases in Flask

This Flask web application demonstrates four machine learning use cases in the Health domain and implements linear regression for predicting insurance charges.

## Features

- **Home Page**: Introduction to the application
- **ML Use Cases**: Four different machine learning applications in healthcare
  - Predicting Insurance Charges (Regression)
  - BMI Prediction for Health Assessment (Regression)
  - Smoking Status Classification (Classification)
  - Family Size Prediction (Regression)
- **Supervised Machine Learning**: Focus on linear regression
  - Basic Concepts explanation
  - Application with real dataset

## Dataset

The application uses the `insurance.csv` dataset which contains information about:
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region
- Insurance charges

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## Usage

1. Navigate to `http://127.0.0.1:5000`
2. Explore the different sections using the navigation menu
3. In the Linear Regression Application, use the prediction form to estimate insurance charges

## Branches

- `main`: Main application code
- `use-cases`: Development of ML use cases
- `linear-regression`: Linear regression implementation

## Technologies Used

- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Bootstrap

## License

This project is for educational purposes.