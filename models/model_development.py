import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.datasets import make_classification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGISTIC_FILE = os.path.join(BASE_DIR, os.pardir, 'data', 'dataset_regresion_logistica.csv')


def _encode_plot(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _train_test_split_linear():
    data = {
        'Study Hours': [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
        'Final Grade': [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
    }
    df = pd.DataFrame(data)
    X = df[['Study Hours']]
    y = df['Final Grade']
    return train_test_split(X, y, test_size=0.25, random_state=42)


def _load_logistic_dataset():
    df = pd.read_csv(LOGISTIC_FILE)
    X = df[[
        'edad',
        'ingreso_mensual',
        'visitas_web_mes',
        'tiempo_sitio_min',
        'compras_previas',
        'descuento_usado'
    ]]
    y = df['target']
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def _generate_random_forest_dataset():
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.03,
        class_sep=1.2,
        random_state=42
    )
    X[:, 0] = np.clip((X[:, 0] * 5 + 30), 1, 100)
    X[:, 1] = np.clip((X[:, 1] * 3 + 10), 1, 50)
    X[:, 2] = np.clip((X[:, 2] * 4 + 20), 0, 100)
    X[:, 3] = np.clip((X[:, 3] * 1.2 + 3), 1, 5)
    X[:, 4] = np.clip((X[:, 4] * 1.1 + 3), 1, 5)
    X[:, 5] = np.clip((X[:, 5] * 1.0 + 3), 1, 5)
    feature_names = [
        'response_time',
        'resolution_attempts',
        'previous_issues',
        'quality_score',
        'staff_rating',
        'complexity'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['response_time'] = df['response_time'].astype(int)
    df['resolution_attempts'] = df['resolution_attempts'].astype(int)
    df['previous_issues'] = df['previous_issues'].astype(int)
    df['quality_score'] = df['quality_score'].round(2)
    df['staff_rating'] = df['staff_rating'].round(2)
    df['complexity'] = df['complexity'].astype(int)
    return train_test_split(df, y, test_size=0.25, random_state=42, stratify=y)


def _build_linear_plots(X_train, y_train, X_test, y_test, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train, y_train, color='blue', label='Train data', s=80, alpha=0.6)
    ax.scatter(X_test, y_test, color='orange', label='Test data', s=80, alpha=0.6)
    x_min = X_train.iloc[:, 0].min()
    x_max = X_train.iloc[:, 0].max()
    x_range = np.linspace(x_min, x_max, 100)
    x_range_df = pd.DataFrame(x_range, columns=['Study Hours'])
    y_pred_line = model.predict(x_range_df)
    ax.plot(x_range, y_pred_line, color='red', linewidth=2, label='Regression line')
    ax.set_title('Linear Regression Training and Test Set', fontsize=14, fontweight='bold')
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Final Grade')
    ax.legend()
    ax.grid(alpha=0.3)
    regression_plot = _encode_plot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = y_test - model.predict(X_test)
    ax.scatter(model.predict(X_test), residuals, color='#6c5ce7', alpha=0.7)
    ax.hlines(0, xmin=model.predict(X_test).min(), xmax=model.predict(X_test).max(), colors='red', linestyles='--')
    ax.set_title('Residuals for Test Set - Linear Regression', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Grade')
    ax.set_ylabel('Residuals')
    ax.grid(alpha=0.3)
    residual_plot = _encode_plot(fig)

    return regression_plot, residual_plot


def _build_logistic_plots(model, X_test_scaled, y_test, y_pred_proba):
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    classes = ['No Purchase', 'Purchase']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
    confusion_plot = _encode_plot(fig)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    roc_plot = _encode_plot(fig)

    return confusion_plot, roc_plot


def _build_random_forest_plots(model, X_test, y_test, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    ax.bar([feature_names[i] for i in indices], importances[indices], color='#0984e3')
    ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.set_ylabel('Importance')
    ax.set_xlabel('Feature')
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    importance_plot = _encode_plot(fig)

    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    ax.figure.colorbar(im, ax=ax)
    classes = ['Class 0', 'Class 1']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
    confusion_plot = _encode_plot(fig)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#00b894', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    roc_plot = _encode_plot(fig)

    return importance_plot, confusion_plot, roc_plot


def _build_prediction_examples(df, y, X, y_pred, prediction_labels=None):
    samples = []
    for index in range(min(3, len(X))):
        row = X.iloc[index].to_dict()
        samples.append({
            'input': row,
            'actual': int(y.iloc[index]) if hasattr(y.iloc[index], 'item') else float(y.iloc[index]),
            'predicted': int(y_pred[index]) if prediction_labels is None else prediction_labels[index],
        })
    return samples


def train_linear_regression_model():
    X_train, X_test, y_train, y_test = _train_test_split_linear()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
        'mae': round(mean_absolute_error(y_test, y_test_pred), 4),
        'r2': round(r2_score(y_test, y_test_pred), 4)
    }
    regression_plot, residual_plot = _build_linear_plots(X_train, y_train, X_test, y_test, model)
    summary = {
        'coefficients': round(model.coef_[0], 4),
        'intercept': round(model.intercept_, 4),
        'equation': f'Final Grade = {round(model.intercept_, 4)} + {round(model.coef_[0], 4)} x Study Hours'
    }
    sample_rows = _build_prediction_examples(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), pd.concat([X_test, X_test]).reset_index(drop=True), y_test_pred)
    return {
        'key': 'linear_regression',
        'display_name': 'Linear Regression',
        'model_type': 'Regression',
        'brief': 'Predicts student final grade based on study hours.',
        'dataset': 'Synthetic student performance dataset',
        'features': ['Study Hours'],
        'target': 'Final Grade',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'hyperparameters': {'fit_intercept': True},
        'metrics': metrics,
        'summary': summary,
        'plots': {
            'regression': regression_plot,
            'residuals': residual_plot
        },
        'console_examples': [
            {
                'input': {'Study Hours': int(X_test.iloc[i].iloc[0])},
                'actual': float(y_test.iloc[i]),
                'predicted': float(round(y_test_pred[i], 2))
            }
            for i in range(min(3, len(X_test)))
        ]
    }


def train_logistic_regression_model():
    X_train, X_test, y_train, y_test = _load_logistic_dataset()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_test_pred), 4),
        'precision': round(precision_score(y_test, y_test_pred), 4),
        'recall': round(recall_score(y_test, y_test_pred), 4),
        'f1': round(f1_score(y_test, y_test_pred), 4),
        'auc': round(roc_auc_score(y_test, y_test_proba), 4)
    }
    confusion_plot, roc_plot = _build_logistic_plots(model, X_test_scaled, y_test, y_test_proba)
    summary = {
        'coefficients': {
            feature: round(coef, 4)
            for feature, coef in zip([
                'Age', 'Monthly Income', 'Web Visits', 'Time on Site', 'Previous Purchases', 'Discount Used'
            ], model.coef_[0])
        },
        'intercept': round(model.intercept_[0], 4)
    }
    console_examples = []
    for i in range(min(3, len(X_test))):
        console_examples.append({
            'input': X_test.iloc[i].to_dict(),
            'actual': int(y_test.iloc[i]),
            'predicted': int(y_test_pred[i]),
            'probability': round(y_test_proba[i] * 100, 2)
        })
    return {
        'key': 'logistic_regression',
        'display_name': 'Logistic Regression',
        'model_type': 'Classification',
        'brief': 'Predicts customer purchase behavior from site usage metrics.',
        'dataset': 'Customer purchase intent dataset',
        'features': ['Age', 'Monthly Income', 'Web Visits', 'Time on Site', 'Previous Purchases', 'Discount Used'],
        'target': 'Purchase (0/1)',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'hyperparameters': {'C': 1.0, 'max_iter': 5000, 'solver': 'lbfgs'},
        'metrics': metrics,
        'summary': summary,
        'plots': {
            'confusion_matrix': confusion_plot,
            'roc_curve': roc_plot
        },
        'console_examples': console_examples
    }


def train_random_forest_model():
    X, X_test, y, y_test = _generate_random_forest_dataset()
    feature_names = X.columns.tolist()
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_test_pred), 4),
        'precision': round(precision_score(y_test, y_test_pred), 4),
        'recall': round(recall_score(y_test, y_test_pred), 4),
        'f1': round(f1_score(y_test, y_test_pred), 4),
        'auc': round(roc_auc_score(y_test, y_test_proba), 4)
    }
    importance_plot, confusion_plot, roc_plot = _build_random_forest_plots(model, X_test, y_test, feature_names)
    summary = {
        'feature_importance': {
            feature: float(round(value, 4))
            for feature, value in zip(feature_names, model.feature_importances_)
        }
    }
    console_examples = []
    for i in range(min(3, len(X_test))):
        console_examples.append({
            'input': X_test.iloc[i].to_dict(),
            'actual': int(y_test[i]),
            'predicted': int(y_test_pred[i]),
            'probability': round(y_test_proba[i] * 100, 2)
        })
    return {
        'key': 'random_forest',
        'display_name': 'Random Forest Classifier',
        'model_type': 'Classification',
        'brief': 'Classifies support outcomes using ensemble tree learning.',
        'dataset': 'Synthetic support satisfaction dataset',
        'features': feature_names,
        'target': 'Satisfaction (0/1)',
        'train_size': len(X),
        'test_size': len(X_test),
        'hyperparameters': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42},
        'metrics': metrics,
        'summary': summary,
        'plots': {
            'feature_importance': importance_plot,
            'confusion_matrix': confusion_plot,
            'roc_curve': roc_plot
        },
        'console_examples': console_examples
    }

MODEL_REPORTS = {
    'linear_regression': train_linear_regression_model(),
    'logistic_regression': train_logistic_regression_model(),
    'random_forest': train_random_forest_model()
}


def get_model_overview():
    return [
        {
            'key': report['key'],
            'display_name': report['display_name'],
            'brief': report['brief'],
            'model_type': report['model_type'],
            'primary_metric': 'RMSE' if report['model_type'] == 'Regression' else 'Accuracy',
            'route': f"/model-development/{report['key']}"
        }
        for report in MODEL_REPORTS.values()
    ]


def get_model_report(key):
    return MODEL_REPORTS.get(key)


def get_comparison_table():
    rows = []
    for report in MODEL_REPORTS.values():
        rows.append({
            'model': report['display_name'],
            'type': report['model_type'],
            'rmse': report['metrics'].get('rmse', 'N/A'),
            'accuracy': report['metrics'].get('accuracy', 'N/A'),
            'f1': report['metrics'].get('f1', 'N/A'),
            'auc': report['metrics'].get('auc', 'N/A'),
            'r2': report['metrics'].get('r2', 'N/A')
        })
    return rows


def get_console_examples():
    return {
        report['key']: report['console_examples']
        for report in MODEL_REPORTS.values()
    }
