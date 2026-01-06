"""
Advanced Clustering Module

This module provides advanced clustering algorithms beyond K-means for customer segmentation.
Includes DBSCAN, HDBSCAN, and Gaussian Mixture Models.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. Install with: pip install hdbscan")


class AdvancedClustering:
    """
    Advanced clustering algorithms for customer segmentation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize advanced clustering.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def dbscan_clustering(self, X: np.ndarray, eps: float = 0.5, 
                        min_samples: int = 5) -> Dict:
        """
        Perform DBSCAN clustering.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in neighborhood
            
        Returns:
            Dict: Clustering results
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Calculate metrics (only if more than 1 cluster found)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        metrics = {}
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        results = {
            'algorithm': 'DBSCAN',
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': list(labels).count(-1),
            'parameters': {'eps': eps, 'min_samples': min_samples},
            'metrics': metrics
        }
        
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = results
        
        return results
    
    def hdbscan_clustering(self, X: np.ndarray, min_cluster_size: int = 10,
                         min_samples: int = 5) -> Dict:
        """
        Perform HDBSCAN clustering.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            min_cluster_size (int): Minimum cluster size
            min_samples (int): Minimum samples in neighborhood
            
        Returns:
            Dict: Clustering results
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(X)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        metrics = {}
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        results = {
            'algorithm': 'HDBSCAN',
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': list(labels).count(-1),
            'parameters': {'min_cluster_size': min_cluster_size, 'min_samples': min_samples},
            'metrics': metrics
        }
        
        self.models['hdbscan'] = clusterer
        self.results['hdbscan'] = results
        
        return results
    
    def gaussian_mixture_clustering(self, X: np.ndarray, n_components: int = 3,
                               covariance_type: str = 'full') -> Dict:
        """
        Perform Gaussian Mixture Model clustering.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance parameters
            
        Returns:
            Dict: Clustering results
        """
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=self.random_state
        )
        labels = gmm.fit_predict(X)
        
        # Calculate metrics
        metrics = {}
        if n_components > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        # Calculate BIC and AIC
        metrics['bic'] = gmm.bic(X)
        metrics['aic'] = gmm.aic(X)
        
        results = {
            'algorithm': 'GaussianMixture',
            'labels': labels,
            'n_clusters': n_components,
            'parameters': {
                'n_components': n_components,
                'covariance_type': covariance_type
            },
            'metrics': metrics,
            'model': gmm
        }
        
        self.models['gaussian_mixture'] = gmm
        self.results['gaussian_mixture'] = results
        
        return results
    
    def hierarchical_clustering(self, X: np.ndarray, n_clusters: int = 3,
                            linkage: str = 'ward') -> Dict:
        """
        Perform hierarchical clustering.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            n_clusters (int): Number of clusters
            linkage (str): Linkage criterion
            
        Returns:
            Dict: Clustering results
        """
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = hierarchical.fit_predict(X)
        
        # Calculate metrics
        metrics = {}
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        results = {
            'algorithm': 'Hierarchical',
            'labels': labels,
            'n_clusters': n_clusters,
            'parameters': {'n_clusters': n_clusters, 'linkage': linkage},
            'metrics': metrics
        }
        
        self.models['hierarchical'] = hierarchical
        self.results['hierarchical'] = results
        
        return results
    
    def compare_algorithms(self, X: np.ndarray, 
                         algorithms: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple clustering algorithms.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            algorithms (List[str]): List of algorithms to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if algorithms is None:
            algorithms = ['dbscan', 'hierarchical', 'gaussian_mixture']
            if HDBSCAN_AVAILABLE:
                algorithms.append('hdbscan')
        
        comparison_results = []
        
        for algorithm in algorithms:
            try:
                if algorithm == 'dbscan':
                    result = self.dbscan_clustering(X)
                elif algorithm == 'hdbscan':
                    result = self.hdbscan_clustering(X)
                elif algorithm == 'gaussian_mixture':
                    result = self.gaussian_mixture_clustering(X)
                elif algorithm == 'hierarchical':
                    result = self.hierarchical_clustering(X)
                else:
                    continue
                
                comparison_results.append({
                    'Algorithm': result['algorithm'],
                    'Clusters': result['n_clusters'],
                    'Silhouette': result['metrics']['silhouette'],
                    'Davies-Bouldin': result['metrics']['davies_bouldin'],
                    'Calinski-Harabasz': result['metrics']['calinski_harabasz']
                })
                
            except Exception as e:
                print(f"Error with {algorithm}: {e}")
                continue
        
        return pd.DataFrame(comparison_results)
    
    def optimize_dbscan(self, X: np.ndarray, eps_range: Tuple[float, float] = (0.1, 2.0),
                        min_samples_range: Tuple[int, int] = (3, 10)) -> Dict:
        """
        Optimize DBSCAN parameters.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            eps_range (Tuple[float, float]): Range for eps parameter
            min_samples_range (Tuple[int, int]): Range for min_samples
            
        Returns:
            Dict: Best parameters and results
        """
        best_score = -1
        best_params = {}
        best_result = None
        
        eps_values = np.linspace(eps_range[0], eps_range[1], 10)
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    result = self.dbscan_clustering(X, eps=eps, min_samples=min_samples)
                    
                    if result['n_clusters'] > 1 and result['n_clusters'] < len(X) // 2:
                        score = result['metrics']['silhouette']
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                            best_result = result
                except:
                    continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result
        }
    
    def get_best_algorithm(self, X: np.ndarray, 
                        metric: str = 'silhouette') -> Dict:
        """
        Get the best performing algorithm based on specified metric.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            metric (str): Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            Dict: Best algorithm results
        """
        comparison = self.compare_algorithms(X)
        
        if metric == 'silhouette':
            best_idx = comparison['Silhouette'].idxmax()
        elif metric == 'davies_bouldin':
            best_idx = comparison['Davies-Bouldin'].idxmin()
        elif metric == 'calinski_harabasz':
            best_idx = comparison['Calinski-Harabasz'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_algorithm = comparison.loc[best_idx, 'Algorithm']
        
        return {
            'best_algorithm': best_algorithm,
            'best_score': comparison.loc[best_idx, metric.title()],
            'comparison_table': comparison
        }


def advanced_segmentation_pipeline(df: pd.DataFrame, 
                              feature_columns: List[str],
                              algorithms: List[str] = None) -> Dict:
    """
    Complete advanced segmentation pipeline.
    
    Args:
        df (pd.DataFrame): Customer features dataframe
        feature_columns (List[str]): Columns to use for clustering
        algorithms (List[str]): Algorithms to try
        
    Returns:
        Dict: Complete segmentation results
    """
    # Scale features
    scaler = StandardScaler()
    X = df[feature_columns].values
    X_scaled = scaler.fit_transform(X)
    
    # Initialize advanced clustering
    clusterer = AdvancedClustering()
    
    # Compare algorithms
    comparison = clusterer.compare_algorithms(X_scaled, algorithms)
    
    # Get best algorithm
    best_result = clusterer.get_best_algorithm(X_scaled)
    
    # Add best labels to dataframe
    best_algorithm_name = best_result['best_algorithm'].lower()
    if best_algorithm_name in clusterer.results:
        best_labels = clusterer.results[best_algorithm_name]['labels']
        df_with_segments = df.copy()
        df_with_segments['Advanced_Cluster'] = best_labels
    else:
        df_with_segments = df.copy()
    
    return {
        'comparison': comparison,
        'best_algorithm': best_result,
        'segmented_data': df_with_segments,
        'scaler': scaler,
        'clusterer': clusterer
    }


if __name__ == "__main__":
    # Example usage
    print("Advanced Clustering Module")
    print("=" * 30)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'CustomerID': [f'C{i:03d}' for i in range(100)],
        'Recency': np.random.normal(100, 30, 100),
        'Frequency': np.random.exponential(3, 100),
        'Monetary': np.random.lognormal(4, 0.5, 100)
    })
    
    # Ensure positive values
    for col in ['Recency', 'Frequency', 'Monetary']:
        sample_data[col] = np.abs(sample_data[col])
    
    # Run advanced clustering
    feature_columns = ['Recency', 'Frequency', 'Monetary']
    results = advanced_segmentation_pipeline(sample_data, feature_columns)
    
    print("Algorithm Comparison:")
    print(results['comparison'])
    print(f"\nBest Algorithm: {results['best_algorithm']['best_algorithm']}")
    print(f"Best Score: {results['best_algorithm']['best_score']:.3f}")
