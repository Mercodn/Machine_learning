"""
K-Means Clustering Model
Unsupervised learning module for customer segmentation
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import base64
from io import BytesIO

# ============================================================================
# DATASET GENERATION (1000+ records, 2 features)
# ============================================================================

def generate_clustering_dataset():
    """Generate synthetic customer spending dataset for clustering"""
    np.random.seed(42)
    
    # Create 1000 customer records with monthly spending and product diversity
    # Feature 1: Monthly Spending (dollars, range 100-10000)
    # Feature 2: Product Diversity (number of different products purchased, range 1-50)
    
    # Generate clusters with different characteristics
    cluster1 = np.random.randn(350, 2) * [1500, 8] + [2000, 15]  # Budget shoppers
    cluster2 = np.random.randn(350, 2) * [2000, 10] + [5000, 30]  # Regular shoppers
    cluster3 = np.random.randn(300, 2) * [1800, 5] + [8000, 40]   # Premium shoppers
    
    # Combine clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Ensure values are within reasonable ranges
    X[:, 0] = np.clip(X[:, 0], 100, 10000)   # Monthly spending: $100-$10,000
    X[:, 1] = np.clip(X[:, 1], 1, 50)        # Product diversity: 1-50 products
    
    # Create dataframe
    df = pd.DataFrame(X, columns=['Monthly_Spending', 'Product_Diversity'])
    df['Monthly_Spending'] = df['Monthly_Spending'].astype(int)
    df['Product_Diversity'] = df['Product_Diversity'].astype(int)
    
    return df, X


# Get dataset and fit model
_df_clustering, _X_clustering = generate_clustering_dataset()
_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_clustering)
_kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
_cluster_labels = _kmeans_model.fit_predict(_X_scaled)
_centroids_scaled = _kmeans_model.cluster_centers_
_centroids_original = _scaler.inverse_transform(_centroids_scaled)


# ============================================================================
# MANUAL K-MEANS SIMULATION DATA (Part 1 - 100 records, 3 iterations)
# ============================================================================

def get_manual_kmeans_data():
    """Get the manual K-Means simulation data with 100 records and 3 iterations"""
    
    # Create 100-record dataset for manual exercise
    np.random.seed(123)
    manual_data = np.random.randn(100, 2) * [2000, 10] + [5000, 25]
    manual_data[:, 0] = np.clip(manual_data[:, 0], 500, 9500)
    manual_data[:, 1] = np.clip(manual_data[:, 1], 2, 50)
    
    manual_df = pd.DataFrame(manual_data, columns=['Spending', 'Diversity'])
    manual_df['Spending'] = manual_df['Spending'].astype(int)
    manual_df['Diversity'] = manual_df['Diversity'].astype(int)
    
    return manual_df


def calculate_euclidean_distance(point, centroid):
    """Calculate Euclidean distance between a point and a centroid"""
    return np.sqrt(np.sum((point - centroid) ** 2))


def perform_manual_kmeans_iteration(data, centroids, iteration_num):
    """
    Perform one iteration of K-Means manually
    Returns: distances table, assignments, new centroids, variance
    """
    n_points = len(data)
    n_clusters = len(centroids)
    
    # Step 1: Calculate distances from each point to each centroid
    distances = np.zeros((n_points, n_clusters))
    for i, point in enumerate(data):
        for j, centroid in enumerate(centroids):
            distances[i, j] = calculate_euclidean_distance(point, centroid)
    
    # Step 2: Assign each point to nearest centroid
    assignments = np.argmin(distances, axis=1)
    
    # Step 3: Calculate new centroids
    new_centroids = np.zeros_like(centroids)
    for k in range(n_clusters):
        cluster_points = data[assignments == k]
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
        else:
            new_centroids[k] = centroids[k]  # Keep old centroid if cluster is empty
    
    # Step 4: Calculate within-cluster variance
    variance = 0
    for k in range(n_clusters):
        cluster_points = data[assignments == k]
        if len(cluster_points) > 0:
            distances_to_centroid = np.sum((cluster_points - new_centroids[k]) ** 2)
            variance += distances_to_centroid
    
    return distances, assignments, new_centroids, variance


def get_manual_kmeans_full_simulation():
    """Get complete manual K-Means simulation with 3 iterations"""
    data = get_manual_kmeans_data().values
    
    # Initial centroids (manually selected)
    initial_centroids = np.array([
        [1500, 10],
        [5000, 25],
        [8500, 40]
    ])
    
    iterations_data = []
    current_centroids = initial_centroids.copy()
    variances = []
    
    for iteration in range(3):
        distances, assignments, new_centroids, variance = perform_manual_kmeans_iteration(
            data, current_centroids, iteration + 1
        )
        
        iterations_data.append({
            'iteration': iteration + 1,
            'centroids': current_centroids.copy(),
            'distances': distances,
            'assignments': assignments,
            'new_centroids': new_centroids.copy(),
            'variance': variance
        })
        
        variances.append(variance)
        current_centroids = new_centroids.copy()
    
    return {
        'data': data,
        'initial_centroids': initial_centroids,
        'iterations': iterations_data,
        'variances': variances,
        'final_centroids': current_centroids,
        'final_assignments': iterations_data[-1]['assignments']
    }


# ============================================================================
# CLUSTERING APPLICATION FUNCTIONS
# ============================================================================

def get_dataset_stats():
    """Get statistics about the clustering dataset"""
    return {
        'total_samples': len(_df_clustering),
        'n_features': 2,
        'n_clusters': 3,
        'features': ['Monthly Spending ($)', 'Product Diversity (count)'],
        'feature_ranges': [
            f"${_df_clustering['Monthly_Spending'].min():.0f} - ${_df_clustering['Monthly_Spending'].max():.0f}",
            f"{_df_clustering['Product_Diversity'].min()} - {_df_clustering['Product_Diversity'].max()}"
        ]
    }


def get_cluster_summary():
    """Get summary statistics for each cluster"""
    df = _df_clustering.copy()
    df['Cluster'] = _cluster_labels
    
    summary = []
    for cluster_id in range(3):
        cluster_data = df[df['Cluster'] == cluster_id]
        summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': round(len(cluster_data) / len(df) * 100, 1),
            'avg_spending': round(cluster_data['Monthly_Spending'].mean(), 2),
            'avg_diversity': round(cluster_data['Product_Diversity'].mean(), 2),
            'min_spending': cluster_data['Monthly_Spending'].min(),
            'max_spending': cluster_data['Monthly_Spending'].max(),
            'min_diversity': cluster_data['Product_Diversity'].min(),
            'max_diversity': cluster_data['Product_Diversity'].max()
        })
    
    return summary


def get_centroids_data():
    """Get centroids in original scale with labels"""
    return [
        {'cluster': 0, 'spending': round(_centroids_original[0, 0], 2), 'diversity': round(_centroids_original[0, 1], 2)},
        {'cluster': 1, 'spending': round(_centroids_original[1, 0], 2), 'diversity': round(_centroids_original[1, 1], 2)},
        {'cluster': 2, 'spending': round(_centroids_original[2, 0], 2), 'diversity': round(_centroids_original[2, 1], 2)}
    ]


def get_cluster_assignments_table(limit=50):
    """Get table of first N data points with their cluster assignments"""
    df = _df_clustering.copy()
    df['Cluster'] = _cluster_labels
    df['Cluster_Label'] = df['Cluster'].map({0: 'Budget', 1: 'Regular', 2: 'Premium'})
    
    return df[['Monthly_Spending', 'Product_Diversity', 'Cluster_Label']].head(limit).to_dict('records')


def get_silhouette_score():
    """Get silhouette score for cluster quality"""
    score = silhouette_score(_X_scaled, _cluster_labels)
    return round(score, 4)


def get_clustering_plot():
    """Generate scatter plot with clusters and centroids"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot points colored by cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for cluster_id in range(3):
        cluster_mask = _cluster_labels == cluster_id
        ax.scatter(_X_clustering[cluster_mask, 0], _X_clustering[cluster_mask, 1],
                  c=colors[cluster_id], label=f'Cluster {cluster_id + 1}',
                  alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Plot centroids
    ax.scatter(_centroids_original[:, 0], _centroids_original[:, 1],
              c='black', marker='*', s=800, edgecolors='yellow', linewidth=2,
              label='Centroids', zorder=5)
    
    ax.set_xlabel('Monthly Spending ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Product Diversity (count)', fontsize=12, fontweight='bold')
    ax.set_title('K-Means Clustering Results (3 Clusters)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def get_inertia_plot():
    """Generate elbow plot showing inertia for different K values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(_X_scaled)
        inertias.append(kmeans.inertia_)
    
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Selected K=3')
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_range)
    
    # Convert to base64
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def get_cluster_interpretation():
    """Get interpretation of clusters"""
    summary = get_cluster_summary()
    
    interpretations = {
        0: {
            'name': 'Budget Shoppers',
            'description': 'Customers with lower spending but moderate product diversity',
            'characteristics': f"Average spending: ${summary[0]['avg_spending']:.0f}, Average products: {summary[0]['avg_diversity']:.1f}",
            'strategy': 'Focus on value offerings, bulk discounts, and promotional campaigns'
        },
        1: {
            'name': 'Regular Shoppers',
            'description': 'Customers with moderate spending and good product diversity',
            'characteristics': f"Average spending: ${summary[1]['avg_spending']:.0f}, Average products: {summary[1]['avg_diversity']:.1f}",
            'strategy': 'Maintain loyalty with personalized recommendations and membership benefits'
        },
        2: {
            'name': 'Premium Shoppers',
            'description': 'High-value customers with highest spending and product diversity',
            'characteristics': f"Average spending: ${summary[2]['avg_spending']:.0f}, Average products: {summary[2]['avg_diversity']:.1f}",
            'strategy': 'Exclusive access, VIP services, and premium product lines'
        }
    }
    
    return interpretations
