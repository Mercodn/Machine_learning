import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('insurance.csv')
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data['sex'] = le_sex.fit_transform(data['sex'])
data['smoker'] = le_smoker.fit_transform(data['smoker'])
data['region'] = le_region.fit_transform(data['region'])

X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Create plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges')
plt.savefig('static/regression_plot.png')
plt.close()