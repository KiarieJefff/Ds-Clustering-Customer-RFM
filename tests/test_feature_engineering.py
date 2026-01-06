"""
Unit Tests for Feature Engineering Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from feature_engineering import (
    calculate_total_price, prepare_date_column, calculate_rfm_features,
    calculate_additional_features, create_customer_features,
    scale_features, get_default_feature_columns, get_rfm_feature_columns
)


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'CustomerID': ['C001', 'C001', 'C002', 'C002', 'C003'],
            'InvoiceDate': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20'],
            'UnitPrice': [10.99, 5.99, 12.99, 8.99, 15.99],
            'Quantity': [2, 3, 1, 4, 2],
            'StockCode': ['A001', 'A002', 'A001', 'A003', 'A002'],
            'Country': ['UK', 'UK', 'US', 'US', 'UK']
        })
    
    def test_calculate_total_price(self):
        """Test total price calculation."""
        df_with_total = calculate_total_price(self.test_data)
        
        # Check TotalPrice column exists
        self.assertIn('TotalPrice', df_with_total.columns)
        
        # Check calculation
        expected_total = self.test_data['UnitPrice'] * self.test_data['Quantity']
        pd.testing.assert_series_equal(df_with_total['TotalPrice'], expected_total)
    
    def test_prepare_date_column(self):
        """Test date column preparation."""
        df_with_date = prepare_date_column(self.test_data)
        
        # Check date type
        self.assertEqual(str(df_with_date['InvoiceDate'].dtype), 'datetime64[ns]')
        
        # Check no null dates
        self.assertFalse(df_with_date['InvoiceDate'].isnull().any())
    
    def test_calculate_rfm_features(self):
        """Test RFM feature calculation."""
        # Prepare data with dates
        df_prepared = prepare_date_column(self.test_data)
        df_prepared = calculate_total_price(df_prepared)
        
        rfm_features = calculate_rfm_features(df_prepared)
        
        # Check required columns exist
        required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                         'TotalItems', 'UniqueProducts', 'Country']
        for col in required_columns:
            self.assertIn(col, rfm_features.columns)
        
        # Check data types
        self.assertEqual(len(rfm_features), 3)  # 3 unique customers
        
        # Check RFM logic
        c001_data = rfm_features[rfm_features['CustomerID'] == 'C001'].iloc[0]
        self.assertEqual(c001_data['Frequency'], 2)  # 2 transactions
        self.assertGreater(c001_data['Recency'], 0)  # Positive recency
        self.assertGreater(c001_data['Monetary'], 0)  # Positive monetary
    
    def test_calculate_additional_features(self):
        """Test additional feature calculation."""
        # Create sample RFM data
        rfm_data = pd.DataFrame({
            'CustomerID': ['C001', 'C002'],
            'Frequency': [2, 3],
            'Monetary': [100, 150],
            'TotalItems': [10, 15]
        })
        
        enhanced_data = calculate_additional_features(rfm_data)
        
        # Check new columns exist
        self.assertIn('AvgOrderValue', enhanced_data.columns)
        self.assertIn('ItemsPerOrder', enhanced_data.columns)
        
        # Check calculations
        c001_data = enhanced_data[enhanced_data['CustomerID'] == 'C001'].iloc[0]
        self.assertEqual(c001_data['AvgOrderValue'], 50)  # 100/2
        self.assertEqual(c001_data['ItemsPerOrder'], 5)  # 10/2
    
    def test_create_customer_features(self):
        """Test complete customer features creation."""
        customer_features = create_customer_features(self.test_data)
        
        # Check it's a DataFrame
        self.assertIsInstance(customer_features, pd.DataFrame)
        
        # Check required columns
        required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary',
                         'AvgOrderValue', 'ItemsPerOrder']
        for col in required_columns:
            self.assertIn(col, customer_features.columns)
        
        # Check one row per customer
        self.assertEqual(len(customer_features), 3)  # 3 unique customers
    
    def test_scale_features(self):
        """Test feature scaling."""
        # Create sample features
        features_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaled_data, scaler = scale_features(features_data, ['feature1', 'feature2'])
        
        # Check scaled data shape
        self.assertEqual(scaled_data.shape, features_data.shape)
        
        # Check mean close to 0 and std close to 1
        self.assertAlmostEqual(scaled_data['feature1_scaled'].mean(), 0, places=1)
        self.assertAlmostEqual(scaled_data['feature1_scaled'].std(), 1, places=1)
    
    def test_get_default_feature_columns(self):
        """Test default feature columns function."""
        columns = get_default_feature_columns()
        
        self.assertIsInstance(columns, list)
        self.assertIn('Recency', columns)
        self.assertIn('Frequency', columns)
        self.assertIn('Monetary', columns)
    
    def test_get_rfm_feature_columns(self):
        """Test RFM feature columns function."""
        columns = get_rfm_feature_columns()
        
        self.assertIsInstance(columns, list)
        self.assertEqual(columns, ['Recency', 'Frequency', 'Monetary'])


if __name__ == '__main__':
    unittest.main()
