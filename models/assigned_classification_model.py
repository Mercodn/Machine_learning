"""
Random Forest Classification Model
Predicts customer satisfaction levels based on various features
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc, roc_auc_score)
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Create a synthetic customer satisfaction dataset
np.random.seed(42)
n_samples = 200

data = {
    "response_time": np.random.randint(1, 48, n_samples),  # hours
    "resolution_attempts": np.random.randint(1, 10, n_samples),
    "customer_previous_issues": np.random.randint(0, 15, n_samples),
    "product_quality_score": np.random.uniform(1, 5, n_samples),
    "support_staff_rating": np.random.uniform(1, 5, n_samples),
    "issue_complexity": np.random.randint(1, 5, n_samples),
}

df = pd.DataFrame(data)

# Create target variable: 1 = satisfied, 0 = unsatisfied
# Satisfied if: quick response, few attempts, good quality, good support
df["satisfaction"] = (
    (df["response_time"] < 24).astype(int) +
    (df["resolution_attempts"] <= 2).astype(int) +
    (df["product_quality_score"] > 3).astype(int) +
    (df["support_staff_rating"] > 3.5).astype(int)
) >= 2

df["satisfaction"] = df["satisfaction"].astype(int)

# Define features and target
X = df[[
    "response_time",
    "resolution_attempts",
    "customer_previous_issues",
    "product_quality_score",
    "support_staff_rating",
    "issue_complexity"
]]

y = df["satisfaction"]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, min_samples_split=5)
model.fit(X, y)

# Calculate predictions for evaluation metrics
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

def predict_satisfaction(response_time, resolution_attempts, previous_issues, quality_score, staff_rating, complexity):
    """
    Predict customer satisfaction based on support metrics.
    
    Args:
        response_time (int): Response time in hours (1-48)
        resolution_attempts (int): Number of attempts to resolve (1-10)
        previous_issues (int): Previous issues count (0-15)
        quality_score (float): Product quality score (1-5)
        staff_rating (float): Support staff rating (1-5)
        complexity (int): Issue complexity level (1-5)
        
    Returns:
        dict: Prediction and probability
    """
    if any(x < 0 for x in [response_time, resolution_attempts, previous_issues, complexity]):
        return None
    
    input_data = np.array([[response_time, resolution_attempts, previous_issues, quality_score, staff_rating, complexity]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return {
        "prediction": "Satisfied" if prediction == 1 else "Unsatisfied",
        "probability_unsatisfied": round(probability[0] * 100, 2),
        "probability_satisfied": round(probability[1] * 100, 2),
        "confidence": round(max(probability) * 100, 2)
    }

def get_feature_importance_plot():
    """
    Generate feature importance visualization.
    
    Returns:
        str: Base64 encoded image
    """
    feature_names = ["Response Time", "Resolution Attempts", "Previous Issues", 
                    "Quality Score", "Staff Rating", "Issue Complexity"]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], align='center', color='steelblue')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
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
        "satisfied": int(y.sum()),
        "unsatisfied": len(y) - int(y.sum()),
        "satisfaction_rate": round((y.sum() / len(y)) * 100, 2),
        "n_features": X.shape[1]
    }

def get_feature_importance_dict():
    """
    Get feature importance as dictionary.
    
    Returns:
        dict: Feature names and their importance scores
    """
    feature_names = ["Response Time", "Resolution Attempts", "Previous Issues", 
                    "Quality Score", "Staff Rating", "Issue Complexity"]
    importances = model.feature_importances_
    
    importance_dict = {}
    for feat, imp in zip(feature_names, importances):
        importance_dict[feat] = round(imp, 4)
    
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


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
    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Unsatisfied', 'Satisfied']
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
    ax.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
    
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
    ax.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
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
        "true_negatives": f"Correctly predicted: Unsatisfied ({cm_data['tn']})",
        "false_positives": f"Incorrectly predicted: Satisfied when Unsatisfied ({cm_data['fp']})",
        "false_negatives": f"Incorrectly predicted: Unsatisfied when Satisfied ({cm_data['fn']})",
        "true_positives": f"Correctly predicted: Satisfied ({cm_data['tp']})"
    }
