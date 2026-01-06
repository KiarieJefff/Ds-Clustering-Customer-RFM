"""
Unit Tests for Clustering Module
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import os

from clustering import (
    find_optimal_clusters, perform_kmeans_clustering, create_cluster_profiles,
    define_segment_names, assign_segments, save_model, load_model
)


class TestClustering(unittest.TestCase):
    """Test cases for clustering functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample customer features
        np.random.seed(42)
        self.test_features = pd.DataFrame({
            'CustomerID': [f'C{i:03d}' for i in range(100)],
            'Recency': np.random.normal(100, 50, 100),
            'Frequency': np.random.exponential(5, 100),
            'Monetary': np.random.lognormal(5, 1, 100),
            'Country': ['UK'] * 60 + ['US'] * 40
        })
        
        # Ensure positive values
        self.test_features['Recency'] = np.abs(self.test_features['Recency'])
        self.test_features['Frequency'] = np.abs(self.test_features['Frequency'])
        self.test_features['Monetary'] = np.abs(self.test_features['Monetary'])
    
    def test_find_optimal_clusters(self):
        """Test optimal cluster finding."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.test_features[['Recency', 'Frequency', 'Monetary']])
        
        cluster_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled)
        
        # Check return types
        self.assertIsInstance(cluster_range, list)
        self.assertIsInstance(wcss, list)
        self.assertIsInstance(silhouette_scores, list)
        
        # Check lengths
        self.assertEqual(len(cluster_range), len(wcss))
        self.assertEqual(len(cluster_range), len(silhouette_scores))
        
        # Check WCSS is decreasing
        self.assertTrue(all(wcss[i] >= wcss[i+1] for i in range(len(wcss)-1)))
        
        # Check silhouette scores are valid
        self.assertTrue(all(-1 <= score <= 1 for score in silhouette_scores))
    
    def test_perform_kmeans_clustering(self):
        """Test K-means clustering."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.test_features[['Recency', 'Frequency', 'Monetary']])
        
        # Perform clustering
        kmeans = perform_kmeans_clustering(X_scaled, n_clusters=3)
        
        # Check it's a KMeans object
        self.assertIsInstance(kmeans, KMeans)
        
        # Check cluster assignments
        self.assertEqual(len(kmeans.labels_), len(self.test_features))
        self.assertEqual(len(np.unique(kmeans.labels_)), 3)
    
    def test_create_cluster_profiles(self):
        """Test cluster profile creation."""
        # Create test data with clusters
        test_data = self.test_features.copy()
        test_data['Cluster'] = np.random.choice([0, 1, 2], len(test_data))
        
        profiles = create_cluster_profiles(test_data)
        
        # Check it's a DataFrame
        self.assertIsInstance(profiles, pd.DataFrame)
        
        # Check required columns
        required_columns = ['Recency', 'Frequency', 'Monetary', 'Count']
        for col in required_columns:
            self.assertIn(col, profiles.columns)
        
        # Check number of profiles
        self.assertEqual(len(profiles), 3)  # 3 clusters
        
        # Check sorting by Monetary
        monetary_values = profiles['Monetary'].values
        self.assertTrue(all(monetary_values[i] >= monetary_values[i+1] 
                        for i in range(len(monetary_values)-1)))
    
    def test_define_segment_names(self):
        """Test segment name definition."""
        # Create test cluster profiles
        profiles = pd.DataFrame({
            'Recency': [50, 200, 400],
            'Frequency': [150, 10, 2],
            'Monetary': [100000, 5000, 500],
            'Count': [100, 200, 300]
        }, index=[0, 1, 2])
        
        segment_names = define_segment_names(profiles)
        
        # Check it's a dictionary
        self.assertIsInstance(segment_names, dict)
        
        # Check all clusters have names
        self.assertEqual(len(segment_names), 3)
        
        # Check segment names are strings
        for name in segment_names.values():
            self.assertIsInstance(name, str)
    
    def test_assign_segments(self):
        """Test segment assignment."""
        # Create test data
        test_data = self.test_features.copy()
        test_data['Cluster'] = np.random.choice([0, 1, 2], len(test_data))
        
        segment_names = {0: 'Segment A', 1: 'Segment B', 2: 'Segment C'}
        
        segmented_data = assign_segments(test_data, segment_names)
        
        # Check Segment column exists
        self.assertIn('Segment', segmented_data.columns)
        
        # Check all segments are assigned
        self.assertFalse(segmented_data['Segment'].isnull().any())
        
        # Check segment names match
        for cluster_id, segment_name in segment_names.items():
            cluster_segments = segmented_data[segmented_data['Cluster'] == cluster_id]['Segment']
            self.assertTrue(all(cluster_segments == segment_name))
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create and fit a simple model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.test_features[['Recency', 'Frequency', 'Monetary']])
        kmeans = perform_kmeans_clustering(X_scaled, n_clusters=3)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_model, \
             tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_scaler:
            
            model_path = tmp_model.name
            scaler_path = tmp_scaler.name
        
        try:
            # Save model
            save_model(kmeans, scaler, model_path, scaler_path)
            
            # Check files exist
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(scaler_path))
            
            # Load model
            loaded_kmeans, loaded_scaler = load_model(model_path, scaler_path)
            
            # Check loaded models
            self.assertIsInstance(loaded_kmeans, KMeans)
            self.assertIsInstance(loaded_scaler, StandardScaler)
            
            # Check predictions match
            original_pred = kmeans.predict(X_scaled)
            loaded_pred = loaded_kmeans.predict(X_scaled)
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(scaler_path):
                os.unlink(scaler_path)


class TestClusteringIntegration(unittest.TestCase):
    """Integration tests for clustering pipeline."""
    
    def test_complete_clustering_pipeline(self):
        """Test complete clustering pipeline."""
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'CustomerID': [f'C{i:03d}' for i in range(50)],
            'Recency': np.random.normal(100, 30, 50),
            'Frequency': np.random.exponential(3, 50),
            'Monetary': np.random.lognormal(4, 0.5, 50),
            'TotalItems': np.random.poisson(10, 50),
            'UniqueProducts': np.random.poisson(5, 50),
            'Country': ['UK'] * 30 + ['US'] * 20
        })
        
        # Ensure positive values
        for col in ['Recency', 'Frequency', 'Monetary', 'TotalItems', 'UniqueProducts']:
            test_data[col] = np.abs(test_data[col])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(test_data[['Recency', 'Frequency', 'Monetary']])
        
        # Find optimal clusters
        cluster_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled, max_clusters=5)
        
        # Perform clustering
        optimal_k = 3
        kmeans = perform_kmeans_clustering(X_scaled, optimal_k)
        
        # Add clusters to data
        test_data['Cluster'] = kmeans.labels_
        
        # Create profiles
        profiles = create_cluster_profiles(test_data)
        
        # Define segments
        segment_names = define_segment_names(profiles)
        
        # Assign segments
        segmented_data = assign_segments(test_data, segment_names)
        
        # Validate results
        self.assertEqual(len(segmented_data), 50)
        self.assertIn('Cluster', segmented_data.columns)
        self.assertIn('Segment', segmented_data.columns)
        self.assertEqual(segmented_data['Cluster'].nunique(), optimal_k)
        self.assertEqual(segmented_data['Segment'].nunique(), optimal_k)
        self.assertFalse(segmented_data['Segment'].isnull().any())


if __name__ == '__main__':
    unittest.main()
