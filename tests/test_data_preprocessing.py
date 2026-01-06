"""
Unit Tests for Data Preprocessing Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from data_preprocessing import (
    load_data, handle_missing_values, remove_duplicates, 
    enforce_dtypes, remove_outliers_iqr, filter_viable_orders,
    preprocess_data, generate_new_customer_id
)


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'InvoiceNo': ['12345', '12346', '12347', '12348', '12345'],
            'StockCode': ['A001', 'A002', 'A003', 'A004', 'A001'],
            'Description': ['Product A', 'Product B', None, 'Product D', 'Product A'],
            'Quantity': [10, 5, 8, -2, 10],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-01'],
            'UnitPrice': [10.99, 5.99, 12.99, 15.99, 10.99],
            'CustomerID': ['C001', 'C002', 'C003', 'C004', None],
            'Country': ['UK', 'US', 'UK', 'FR', 'UK']
        })
        
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test data."""
        os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Test data loading function."""
        df = load_data(self.temp_file.name)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn('InvoiceNo', df.columns)
        self.assertIn('CustomerID', df.columns)
    
    def test_generate_new_customer_id(self):
        """Test new customer ID generation."""
        # Generate multiple IDs
        ids = [generate_new_customer_id() for _ in range(10)]
        
        # Check format
        for customer_id in ids:
            self.assertTrue(customer_id.startswith('N'))
            self.assertEqual(len(customer_id), 5)  # N + 4 digits
            self.assertTrue(customer_id[1:].isdigit())  # Check digits after N
            
            # Check range
            number_part = int(customer_id[1:])
            self.assertGreaterEqual(number_part, 1001)
            self.assertLessEqual(number_part, 9999)
        
        # Check uniqueness
        self.assertEqual(len(set(ids)), len(ids))
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_clean = handle_missing_values(self.test_data)
        
        # Check that missing Description is removed
        self.assertEqual(len(df_clean), 4)  # One row with missing Description removed
        
        # Check that missing CustomerID is replaced with NXXXX format
        self.assertFalse(df_clean['CustomerID'].isnull().any())
        
        # Check that new CustomerIDs follow NXXXX pattern
        new_customer_ids = df_clean[df_clean['CustomerID'].str.startswith('N', na=False)]['CustomerID']
        for customer_id in new_customer_ids:
            self.assertTrue(customer_id.startswith('N'))
            self.assertTrue(len(customer_id) == 5)  # N + 4 digits
            self.assertTrue(customer_id[1:].isdigit())  # Check digits after N
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df_no_duplicates = remove_duplicates(self.test_data)
        
        # Should remove one duplicate row
        self.assertEqual(len(df_no_duplicates), 4)
        
        # Check no duplicates remain
        self.assertFalse(df_no_duplicates.duplicated().any())
    
    def test_enforce_dtypes(self):
        """Test data type enforcement."""
        df_typed = enforce_dtypes(self.test_data)
        
        # Check data types
        self.assertEqual(df_typed['InvoiceNo'].dtype, 'object')
        self.assertEqual(df_typed['Quantity'].dtype, 'Int64')
        self.assertEqual(str(df_typed['InvoiceDate'].dtype), 'datetime64[ns]')
        self.assertEqual(df_typed['CustomerID'].dtype, 'object')
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        # Create data with outliers
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, 200]  # 100 and 200 are outliers
        })
        
        df_no_outliers = remove_outliers_iqr(data_with_outliers, ['values'])
        
        # Should remove outliers
        self.assertLess(len(df_no_outliers), len(data_with_outliers))
        self.assertGreater(len(df_no_outliers), 0)
    
    def test_filter_viable_orders(self):
        """Test filtering of viable orders."""
        df_viable = filter_viable_orders(self.test_data)
        
        # Should remove negative quantity orders
        self.assertFalse((df_viable['Quantity'] < 0).any())
        self.assertLess(len(df_viable), len(self.test_data))
    
    def test_preprocess_data_pipeline(self):
        """Test complete preprocessing pipeline."""
        df_processed = preprocess_data(self.temp_file.name)
        
        # Check that data is processed
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertGreater(len(df_processed), 0)
        
        # Check that negative quantities are removed
        self.assertFalse((df_processed['Quantity'] < 0).any())
        
        # Check that missing values are handled
        self.assertFalse(df_processed['Description'].isnull().any())


if __name__ == '__main__':
    unittest.main()
