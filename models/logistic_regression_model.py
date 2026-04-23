"""
Logistic Regression Model for Purchase Prediction
Predicts whether a customer will make a purchase based on behavioral features
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc, roc_auc_score)
import io
import base64

# Load the dataset
df = pd.read_csv("data/dataset_regresion_logistica.csv")

# Define predictor variables (features)
X = df[[
    "edad",              # age
    "ingreso_mensual",   # monthly income
    "visitas_web_mes",   # website visits per month
    "tiempo_sitio_min",  # time spent on site (minutes)
    "compras_previas",   # number of previous purchases
    "descuento_usado"    # discount used (0/1)
]]

# Define target variable
y = df["target"]  # binary: 1 = purchased, 0 = did not purchase

# Train logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X, y)

# Calculate training predictions for evaluation metrics
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

def predict_purchase(age, income, web_visits, time_spent, prev_purchases, discount_used):
    """
    Predict whether a customer will make a purchase.
    
    Args:
        age (int): Customer age
        income (float): Monthly income
        web_visits (int): Website visits per month
        time_spent (float): Time spent on site (minutes)
        prev_purchases (int): Number of previous purchases
        discount_used (int): Whether discount was used (0 or 1)
        
    Returns:
        dict: Prediction and probability
    """
    if any(x < 0 for x in [age, income, web_visits, time_spent, prev_purchases]):
        return None
    
    input_data = np.array([[age, income, web_visits, time_spent, prev_purchases, discount_used]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return {
        "prediction": "Will Purchase" if prediction == 1 else "Won't Purchase",
        "probability_no_purchase": round(probability[0] * 100, 2),
        "probability_purchase": round(probability[1] * 100, 2)
    }

def get_partial_dependence_plot():
    """
    Generate partial dependence plot for web visits feature.
    
    Returns:
        str: Base64 encoded image
    """
    feature_to_plot = "visitas_web_mes"
    x_range = np.linspace(df[feature_to_plot].min(), df[feature_to_plot].max(), 300).reshape(-1, 1)
    
    X_temp = pd.DataFrame({
        "edad": [df["edad"].mean()] * len(x_range),
        "ingreso_mensual": [df["ingreso_mensual"].mean()] * len(x_range),
        "visitas_web_mes": x_range.flatten(),
        "tiempo_sitio_min": [df["tiempo_sitio_min"].mean()] * len(x_range),
        "compras_previas": [df["compras_previas"].mean()] * len(x_range),
        "descuento_usado": [df["descuento_usado"].mean()] * len(x_range)
    })
    
    probabilities = model.predict_proba(X_temp)[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, probabilities, color='green', linewidth=2, label='Purchase Probability')
    ax.fill_between(x_range.flatten(), probabilities, alpha=0.3, color='green')
    ax.set_xlabel('Website Visits per Month', fontsize=12)
    ax.set_ylabel('Probability of Purchase', fontsize=12)
    ax.set_title('Partial Dependence: Purchase Probability vs Web Visits', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
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
        "total_samples": len(df),
        "purchases": int(y.sum()),
        "no_purchases": len(y) - int(y.sum()),
        "purchase_rate": round((y.sum() / len(y)) * 100, 2)
    }

def get_feature_importance():
    """
    Get feature importance (coefficients) from the logistic regression model.
    
    Returns:
        dict: Feature importances
    """
    features = ["Age", "Monthly Income", "Web Visits", "Time on Site", "Previous Purchases", "Discount Used"]
    coef = model.coef_[0]
    
    importance_dict = {}
    for feat, coeff in zip(features, coef):
        importance_dict[feat] = round(coeff, 4)
    
    return importance_dict


# ============================================================================
# EVALUATION METRICS FUNCTIONS
# ============================================================================

def get_confusion_matrix_data():
    """
    Calculate confusion matrix data.
    
    Returns:
        dict: Confusion matrix values and labels
    """
    cm = confusion_matrix(y, y_pred)
    
    return {
        "tn": int(cm[0][0]),   # True Negatives
        "fp": int(cm[0][1]),   # False Positives
        "fn": int(cm[1][0]),   # False Negatives
        "tp": int(cm[1][1]),   # True Positives
        "matrix": cm.tolist()
    }

def get_confusion_matrix_plot():
    """
    Generate confusion matrix visualization.
    
    Returns:
        str: Base64 encoded image
    """
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Labels
    classes = ['No Purchase', 'Purchase']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add values to cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=14, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_classification_metrics():
    """
    Calculate classification evaluation metrics.
    
    Returns:
        dict: All classification metrics
    """
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_score = roc_auc_score(y, y_pred_proba)
    
    return {
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "precision": round(precision, 4),
        "precision_percent": round(precision * 100, 2),
        "recall": round(recall, 4),
        "recall_percent": round(recall * 100, 2),
        "f1_score": round(f1, 4),
        "f1_percent": round(f1 * 100, 2),
        "auc": round(auc_score, 4),
        "auc_percent": round(auc_score * 100, 2)
    }

def get_roc_curve_plot():
    """
    Generate ROC curve visualization.
    
    Returns:
        str: Base64 encoded image
    """
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_metrics_interpretation():
    """
    Get interpretation guidance for metrics.
    
    Returns:
        dict: Interpretation text for each metric
    """
    metrics = get_classification_metrics()
    cm_data = get_confusion_matrix_data()
    
    return {
        "accuracy": "Percentage of correct predictions overall. Ranges from 0 to 1 (0-100%).",
        "precision": "Of positive predictions, how many were correct. Important when false positives are costly.",
        "recall": "Of actual positives, how many were correctly identified. Important when false negatives are costly.",
        "f1_score": "Harmonic mean of precision and recall. Best single metric when balancing both concerns.",
        "auc": "Area Under the ROC Curve. Probability that model ranks random positive higher than random negative.",
        "true_negatives": f"Correctly predicted: No Purchase ({cm_data['tn']})",
        "false_positives": f"Incorrectly predicted: Purchase when No Purchase ({cm_data['fp']})",
        "false_negatives": f"Incorrectly predicted: No Purchase when Purchase ({cm_data['fn']})",
        "true_positives": f"Correctly predicted: Purchase ({cm_data['tp']})"
    }
