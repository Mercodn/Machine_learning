"""
Linear Regression Model for Grade Prediction
Predicts Final Grade based on Study Hours
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import base64

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)
X = df[["Study Hours"]]
y = df["Final Grade"]

# Train the model
model = LinearRegression()
model.fit(X, y)

def predict_grade(hours):
    """
    Predict the final grade based on study hours.
    
    Args:
        hours (float): Number of study hours
        
    Returns:
        float: Predicted final grade
    """
    if hours < 0:
        return None
    result = model.predict([[hours]])[0]
    return round(result, 2)

def get_model_parameters():
    """
    Get the slope and intercept of the regression line.
    
    Returns:
        dict: Contains slope and intercept
    """
    return {
        "slope": round(model.coef_[0], 4),
        "intercept": round(model.intercept_, 4),
        "equation": f"Grade = {round(model.intercept_, 4)} + {round(model.coef_[0], 4)} x Study Hours"
    }

def get_regression_plot():
    """
    Generate a regression plot visualization.
    
    Returns:
        str: Base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(X, y, color='blue', label='Actual Data', s=100, alpha=0.6)
    
    # Regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')
    
    ax.set_xlabel('Study Hours', fontsize=12)
    ax.set_ylabel('Final Grade', fontsize=12)
    ax.set_title('Linear Regression: Grade vs Study Hours', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_dataset_stats():
    """
    Get statistics about the dataset.
    
    Returns:
        dict: Dataset statistics
    """
    return {
        "count": len(df),
        "hours_mean": round(df["Study Hours"].mean(), 2),
        "hours_std": round(df["Study Hours"].std(), 2),
        "grade_mean": round(df["Final Grade"].mean(), 2),
        "grade_std": round(df["Final Grade"].std(), 2),
        "correlation": round(df["Study Hours"].corr(df["Final Grade"]), 4)
    }
