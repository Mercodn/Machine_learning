"""
Machine Learning Web Application
Educational platform showcasing ML concepts with Flask
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import webbrowser
import threading
import time


import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from linear_regression_model import predict_grade, get_model_parameters, get_regression_plot, get_dataset_stats as lr_dataset_stats
from logistic_regression_model import (predict_purchase, get_partial_dependence_plot, get_dataset_stats as log_dataset_stats, 
                                        get_feature_importance, get_confusion_matrix_data, get_confusion_matrix_plot,
                                        get_classification_metrics, get_roc_curve_plot, get_metrics_interpretation)
from assigned_classification_model import (predict_satisfaction, get_feature_importance_plot, 
                                           get_dataset_stats as assigned_dataset_stats,
                                           get_feature_importance_dict, get_confusion_matrix_data as assigned_cm_data,
                                           get_confusion_matrix_plot as assigned_cm_plot, get_classification_metrics as assigned_metrics,
                                           get_roc_curve_plot as assigned_roc, get_metrics_interpretation as assigned_interpretation)
from kmeans_clustering_model import (get_dataset_stats as kmeans_dataset_stats, get_cluster_summary, get_centroids_data,
                                     get_cluster_assignments_table, get_silhouette_score, get_clustering_plot, 
                                     get_inertia_plot, get_cluster_interpretation)

app = Flask(__name__)

def open_browser():
    """Open browser after a short delay to allow Flask to start (development only)"""
    if os.getenv('FLASK_ENV') != 'production':
        time.sleep(1.5)  # Wait for Flask to start
        webbrowser.open('http://127.0.0.1:5000')


@app.route("/")
def home():
    """Display the main home page"""
    return render_template("index.html")


@app.route("/ml-use-cases")
def ml_use_cases():
    """Display overview of all ML use cases"""
    use_cases = [
        {
            "id": 1,
            "title": "Use Case 1: Student Performance Prediction",
            "description": "Predicting student grades based on study time using Linear Regression",
            "type": "Supervised Learning - Regression"
        },
        {
            "id": 2,
            "title": "Use Case 2: Customer Purchase Intent",
            "description": "Predicting customer purchase behavior using Logistic Regression",
            "type": "Supervised Learning - Classification"
        },
        {
            "id": 3,
            "title": "Use Case 3: House Price Estimation",
            "description": "Estimating real estate prices based on property features",
            "type": "Supervised Learning - Regression"
        },
        {
            "id": 4,
            "title": "Use Case 4: Medical Diagnosis Support",
            "description": "Classifying medical conditions based on patient symptoms and tests",
            "type": "Supervised Learning - Classification"
        }
    ]
    return render_template("use_cases.html", use_cases=use_cases)


@app.route("/use-case/<int:case_id>")
def use_case_detail(case_id):
    """Display detailed information for a specific use case"""
    use_cases = {
        1: {
            "title": "Use Case 1: Student Performance Prediction",
            "domain": "Education",
            "problem": "Educational institutions need to predict student academic performance to provide timely interventions.",
            "ml_type": "Supervised Learning - Linear Regression",
            "input": "Study hours per week",
            "output": "Predicted final grade (0-5 scale)",
            "relevance": "Helps educators identify at-risk students and customize learning strategies.",
            "implementation": "Used in the Linear Regression module"
        },
        2: {
            "title": "Use Case 2: Customer Purchase Intent",
            "domain": "E-commerce",
            "problem": "Online retailers need to identify customers likely to make a purchase to optimize marketing efforts.",
            "ml_type": "Supervised Learning - Logistic Regression",
            "input": "Age, income, web visits, time on site, previous purchases, discount usage",
            "output": "Binary prediction (purchase/no purchase) with probability",
            "relevance": "Enables targeted marketing and improves conversion rates.",
            "implementation": "Used in the Logistic Regression module"
        },
        3: {
            "title": "Use Case 3: House Price Estimation",
            "domain": "Real Estate",
            "problem": "Real estate professionals need accurate price predictions to assess property value.",
            "ml_type": "Supervised Learning - Multiple Linear Regression",
            "input": "Square footage, number of rooms, location, age, amenities",
            "output": "Predicted house price",
            "relevance": "Facilitates fair pricing, market analysis, and investment decisions.",
            "implementation": "Example of regression in real-world finance"
        },
        4: {
            "title": "Use Case 4: Medical Diagnosis Support",
            "domain": "Healthcare",
            "problem": "Healthcare providers need diagnostic tools to support clinical decision-making.",
            "ml_type": "Supervised Learning - Logistic Regression / Neural Networks",
            "input": "Patient symptoms, lab results, medical history, vital signs",
            "output": "Disease probability classification",
            "relevance": "Improves diagnostic accuracy and enables early disease detection.",
            "implementation": "Example of binary/multiclass classification in healthcare"
        }
    }
    
    if case_id in use_cases:
        return render_template("use_case_detail.html", case=use_cases[case_id], case_id=case_id)
    return "Use case not found", 404


@app.route("/supervised-learning")
def supervised_learning():
    """Display introduction to supervised learning"""
    return render_template("supervised_learning.html")



@app.route("/linear-regression/concepts")
def lr_concepts():
    """Display Linear Regression basic concepts"""
    parameters = get_model_parameters()
    stats = lr_dataset_stats()
    
    return render_template("lr_concepts.html", 
                         parameters=parameters, 
                         stats=stats)


@app.route("/linear-regression/application")
def lr_application():
    """Display Linear Regression application with prediction form"""
    plot_url = get_regression_plot()
    stats = lr_dataset_stats()
    
    return render_template("lr_application.html",
                         plot_url=plot_url,
                         stats=stats,
                         result=None,
                         parameters=None)


@app.route("/linear-regression/predict", methods=["POST"])
def lr_predict():
    """Handle linear regression prediction"""
    try:
        hours = float(request.form.get("hours", 0))
        
        # Validation
        if hours < 0:
            return render_template("lr_application.html",
                                 error="Study hours must be a positive number",
                                 plot_url=get_regression_plot(),
                                 stats=lr_dataset_stats(),
                                 result=None,
                                 parameters=None)
        
        if hours > 24:
            return render_template("lr_application.html",
                                 error="Study hours cannot exceed 24 hours per day",
                                 plot_url=get_regression_plot(),
                                 stats=lr_dataset_stats(),
                                 result=None,
                                 parameters=None)
        
        result = predict_grade(hours)
        parameters = get_model_parameters()
        
        return render_template("lr_application.html",
                             result=result,
                             hours=hours,
                             plot_url=get_regression_plot(),
                             parameters=parameters,
                             stats=lr_dataset_stats())
    
    except ValueError:
        return render_template("lr_application.html",
                             error="Please enter a valid number",
                             plot_url=get_regression_plot(),
                             stats=lr_dataset_stats(),
                             result=None,
                             parameters=None)

@app.route("/logistic-regression/concepts")
def log_concepts():
    """Display Logistic Regression basic concepts"""
    feature_importance = get_feature_importance()
    stats = log_dataset_stats()
    
    return render_template("log_concepts.html",
                         feature_importance=feature_importance,
                         stats=stats)


@app.route("/logistic-regression/application")
def log_application():
    """Display Logistic Regression application"""
    plot_url = get_partial_dependence_plot()
    stats = log_dataset_stats()
    
    return render_template("log_application.html",
                         plot_url=plot_url,
                         stats=stats,
                         result=None)


@app.route("/logistic-regression/predict", methods=["POST"])
def log_predict():
    """Handle logistic regression prediction"""
    try:
        age = int(request.form.get("age", 0))
        income = float(request.form.get("income", 0))
        web_visits = int(request.form.get("web_visits", 0))
        time_spent = float(request.form.get("time_spent", 0))
        prev_purchases = int(request.form.get("prev_purchases", 0))
        discount_used = int(request.form.get("discount_used", 0))
        
        # Validation
        if any(x < 0 for x in [age, income, web_visits, time_spent, prev_purchases]):
            return render_template("log_application.html",
                                 error="All values must be positive numbers",
                                 plot_url=get_partial_dependence_plot(),
                                 stats=log_dataset_stats(),
                                 result=None)
        
        if age > 150 or age < 13:
            return render_template("log_application.html",
                                 error="Please enter a valid age (13-150)",
                                 plot_url=get_partial_dependence_plot(),
                                 stats=log_dataset_stats(),
                                 result=None)
        
        result = predict_purchase(age, income, web_visits, time_spent, prev_purchases, discount_used)
        
        return render_template("log_application.html",
                             result=result,
                             plot_url=get_partial_dependence_plot(),
                             stats=log_dataset_stats())
    
    except ValueError:
        return render_template("log_application.html",
                             error="Please enter valid numbers",
                             plot_url=get_partial_dependence_plot(),
                             stats=log_dataset_stats(),
                             result=None)




@app.route("/logistic-regression/metrics")
def log_metrics():
    """Display Logistic Regression evaluation metrics"""
    confusion_matrix_plot = get_confusion_matrix_plot()
    metrics = get_classification_metrics()
    cm_data = get_confusion_matrix_data()
    roc_curve_plot = get_roc_curve_plot()
    interpretation = get_metrics_interpretation()
    
    return render_template("log_metrics.html",
                         confusion_matrix_plot=confusion_matrix_plot,
                         metrics=metrics,
                         cm_data=cm_data,
                         roc_curve_plot=roc_curve_plot,
                         interpretation=interpretation)




@app.route("/assigned-model/concepts")
def assigned_concepts():
    """Display assigned classification model concepts"""
    return render_template("assigned_model_concepts.html")


@app.route("/assigned-model/application")
def assigned_application():
    """Display assigned model application"""
    feature_importance_plot = get_feature_importance_plot()
    stats = assigned_dataset_stats()
    
    return render_template("assigned_model_application.html",
                         feature_importance_plot=feature_importance_plot,
                         stats=stats)


@app.route("/assigned-model/predict", methods=["POST"])
def assigned_predict():
    """Handle assigned model prediction"""
    try:
        response_time = int(request.form.get("response_time", 0))
        resolution_attempts = int(request.form.get("resolution_attempts", 0))
        previous_issues = int(request.form.get("previous_issues", 0))
        quality_score = float(request.form.get("quality_score", 0))
        staff_rating = float(request.form.get("staff_rating", 0))
        complexity = int(request.form.get("complexity", 0))
        
        # Validation
        if any(x < 0 for x in [response_time, resolution_attempts, previous_issues, complexity]):
            return render_template("assigned_model_application.html",
                                 error="All values must be positive numbers",
                                 feature_importance_plot=get_feature_importance_plot(),
                                 stats=assigned_dataset_stats())
        
        if response_time > 48 or response_time < 1:
            return render_template("assigned_model_application.html",
                                 error="Response time must be between 1 and 48 hours",
                                 feature_importance_plot=get_feature_importance_plot(),
                                 stats=assigned_dataset_stats())
        
        if quality_score < 1 or quality_score > 5:
            return render_template("assigned_model_application.html",
                                 error="Quality score must be between 1 and 5",
                                 feature_importance_plot=get_feature_importance_plot(),
                                 stats=assigned_dataset_stats())
        
        result = predict_satisfaction(response_time, resolution_attempts, previous_issues, quality_score, staff_rating, complexity)
        
        return render_template("assigned_model_application.html",
                             result=result,
                             feature_importance_plot=get_feature_importance_plot(),
                             stats=assigned_dataset_stats())
    
    except ValueError:
        return render_template("assigned_model_application.html",
                             error="Please enter valid numbers",
                             feature_importance_plot=get_feature_importance_plot(),
                             stats=assigned_dataset_stats())


@app.route("/assigned-model/metrics")
def assigned_model_metrics_view():
    """Display assigned model evaluation metrics"""
    confusion_matrix_plot = assigned_cm_plot()
    metrics = assigned_metrics()
    cm_data = assigned_cm_data()
    roc_curve_plot = assigned_roc()
    interpretation = assigned_interpretation()
    
    return render_template("assigned_model_metrics.html",
                         confusion_matrix_plot=confusion_matrix_plot,
                         metrics=metrics,
                         cm_data=cm_data,
                         roc_curve_plot=roc_curve_plot,
                         interpretation=interpretation)


@app.route("/unsupervised-learning/concepts")
def unsupervised_concepts():
    """Display unsupervised learning and K-Means concepts"""
    return render_template("unsupervised_concepts.html")


@app.route("/unsupervised-learning/manual-exercise")
def manual_kmeans_exercise():
    """Display manual K-Means simulation exercise"""
    return render_template("manual_kmeans_exercise.html")


@app.route("/unsupervised-learning/clustering-app")
def clustering_application():
    """Display K-Means clustering application with real data"""
    stats = kmeans_dataset_stats()
    cluster_summary = get_cluster_summary()
    centroids = get_centroids_data()
    cluster_assignments = list(enumerate(get_cluster_assignments_table(50)))
    silhouette_score = get_silhouette_score()
    clustering_plot = get_clustering_plot()
    inertia_plot = get_inertia_plot()
    interpretation = get_cluster_interpretation()
    
    return render_template("clustering_application.html",
                         stats=stats,
                         cluster_summary=cluster_summary,
                         centroids=centroids,
                         cluster_assignments=cluster_assignments,
                         silhouette_score=silhouette_score,
                         clustering_plot=clustering_plot,
                         inertia_plot=inertia_plot,
                         interpretation=interpretation)



@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500


if __name__ == "__main__":
   
    threading.Thread(target=open_browser).start()


    port = int(os.environ.get('PORT', 5000))

    
    debug_mode = os.getenv('FLASK_ENV') != 'production'

    app.run(debug=debug_mode, host='0.0.0.0', port=port)