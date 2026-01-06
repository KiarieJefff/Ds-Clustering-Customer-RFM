"""
Clustering Module

This module contains functions for customer segmentation using K-means clustering,
including optimal cluster selection and segment profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import joblib


def find_optimal_clusters(X: np.ndarray, 
                         max_clusters: int = 10,
                         min_clusters: int = 2,
                         random_state: int = 42) -> Tuple[List[int], List[float], List[float]]:
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.
    
    Args:
        X (np.ndarray): Scaled feature matrix
        max_clusters (int): Maximum number of clusters to test
        min_clusters (int): Minimum number of clusters to test
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[List[int], List[float], List[float]]: 
            - List of cluster numbers tested
            - List of WCSS values (Elbow Method)
            - List of Silhouette scores
    """
    cluster_range = range(min_clusters, max_clusters + 1)
    wcss = []
    silhouette_scores = []
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    return list(cluster_range), wcss, silhouette_scores


def plot_elbow_method(cluster_range: List[int], wcss: List[float], 
                     save_path: Optional[str] = None) -> None:
    """
    Plot the Elbow Method for optimal cluster selection.
    
    Args:
        cluster_range (List[int]): List of cluster numbers
        wcss (List[float]): List of WCSS values
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, wcss, 'bx-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal Number of Clusters', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_silhouette_scores(cluster_range: List[int], silhouette_scores: List[float],
                          save_path: Optional[str] = None) -> None:
    """
    Plot Silhouette Scores for different cluster numbers.
    
    Args:
        cluster_range (List[int]): List of cluster numbers
        silhouette_scores (List[float]): List of silhouette scores
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Scores for Different Cluster Numbers', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, score in enumerate(silhouette_scores):
        plt.annotate(f'{score:.3f}', 
                    (cluster_range[i], silhouette_scores[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def perform_kmeans_clustering(X: np.ndarray, 
                           n_clusters: int,
                           random_state: int = 42) -> KMeans:
    """
    Perform K-means clustering.
    
    Args:
        X (np.ndarray): Scaled feature matrix
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
        
    Returns:
        KMeans: Fitted K-means model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    return kmeans


def create_cluster_profiles(df: pd.DataFrame, 
                          cluster_col: str = 'Cluster',
                          feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Create cluster profiles by calculating mean values for each cluster.
    
    Args:
        df (pd.DataFrame): Dataframe with cluster assignments
        cluster_col (str): Name of cluster column
        feature_cols (List[str]): List of feature columns to profile
        
    Returns:
        pd.DataFrame: Cluster profiles
    """
    if feature_cols is None:
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'TotalItems', 
                      'UniqueProducts', 'AvgOrderValue', 'ItemsPerOrder']
    
    # Calculate cluster statistics
    cluster_profile = df.groupby(cluster_col).agg({
        **{col: 'mean' for col in feature_cols},
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    
    # Sort clusters by monetary value (descending)
    cluster_profile = cluster_profile.sort_values('Monetary', ascending=False)
    
    return cluster_profile


def define_segment_names(cluster_profile: pd.DataFrame) -> Dict[int, str]:
    """
    Define segment names based on RFM characteristics.
    
    Args:
        cluster_profile (pd.DataFrame): Cluster profiles
        
    Returns:
        Dict[int, str]: Mapping of cluster numbers to segment names
    """
    segment_names = {}
    
    # Sort clusters by monetary value to identify high-value segments
    sorted_clusters = cluster_profile.sort_values('Monetary', ascending=False)
    
    for i, (cluster_id, profile) in enumerate(sorted_clusters.iterrows()):
        recency = profile['Recency']
        frequency = profile['Frequency']
        monetary = profile['Monetary']
        
        # Define segment logic based on RFM characteristics
        if frequency > 100 and monetary > 50000:
            segment_names[cluster_id] = 'Loyal High-Spenders'
        elif recency > 300 and frequency < 5:
            segment_names[cluster_id] = 'At-Risk Customers'
        elif frequency > 10 and monetary > 2000:
            segment_names[cluster_id] = 'Regular Customers'
        elif recency < 100 and frequency > 5:
            segment_names[cluster_id] = 'Recent Engaged Customers'
        elif monetary < 500:
            segment_names[cluster_id] = 'Low-Value Customers'
        else:
            segment_names[cluster_id] = 'Bargain Hunters'
    
    return segment_names


def assign_segments(df: pd.DataFrame, 
                   segment_names: Dict[int, str],
                   cluster_col: str = 'Cluster') -> pd.DataFrame:
    """
    Assign segment names to customers based on cluster assignments.
    
    Args:
        df (pd.DataFrame): Dataframe with cluster assignments
        segment_names (Dict[int, str]): Mapping of cluster numbers to segment names
        cluster_col (str): Name of cluster column
        
    Returns:
        pd.DataFrame: Dataframe with segment assignments
    """
    df = df.copy()
    df['Segment'] = df[cluster_col].map(segment_names)
    return df


def save_model(model: KMeans, scaler: StandardScaler, 
               model_path: str, scaler_path: str) -> None:
    """
    Save the trained model and scaler.
    
    Args:
        model (KMeans): Trained K-means model
        scaler (StandardScaler): Fitted scaler
        model_path (str): Path to save the model
        scaler_path (str): Path to save the scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def load_model(model_path: str, scaler_path: str) -> Tuple[KMeans, StandardScaler]:
    """
    Load the trained model and scaler.
    
    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        
    Returns:
        Tuple[KMeans, StandardScaler]: Loaded model and scaler
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_new_customer(model: KMeans, scaler: StandardScaler, 
                        customer_features: pd.DataFrame) -> np.ndarray:
    """
    Predict cluster for new customer data.
    
    Args:
        model (KMeans): Trained K-means model
        scaler (StandardScaler): Fitted scaler
        customer_features (pd.DataFrame): New customer features
        
    Returns:
        np.ndarray: Predicted cluster labels
    """
    # Scale the features
    scaled_features = scaler.transform(customer_features)
    
    # Predict clusters
    predictions = model.predict(scaled_features)
    
    return predictions


if __name__ == "__main__":
    # Example usage
    features_path = "../data/processed/Customer_RFM_Features.csv"
    
    # Load customer features
    df = pd.read_csv(features_path)
    
    # Select features for clustering
    feature_cols = ['Recency', 'Frequency', 'Monetary']
    X = df[feature_cols].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    cluster_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled)
    
    # Plot results
    plot_elbow_method(cluster_range, wcss)
    plot_silhouette_scores(cluster_range, silhouette_scores)
    
    # Perform clustering with optimal k (e.g., k=3)
    optimal_k = 3
    kmeans = perform_kmeans_clustering(X_scaled, optimal_k)
    
    # Add cluster assignments to dataframe
    df['Cluster'] = kmeans.labels_
    
    # Create cluster profiles
    cluster_profile = create_cluster_profiles(df)
    print("Cluster Profiles:")
    print(cluster_profile)
    
    # Define segment names
    segment_names = define_segment_names(cluster_profile)
    
    # Assign segments
    df = assign_segments(df, segment_names)
    
    # Save results
    df.to_csv("../data/processed/Customer_Segments.csv", index=False)
    
    # Save model and scaler
    save_model(kmeans, scaler, 
               "../models/kmeans_customer_segmentation.pkl",
               "../models/scaler_customer_segmentation.pkl")
    
    print(f"\nClustering completed with {optimal_k} clusters")
    print("Results saved to: ../data/processed/Customer_Segments.csv")
