"""
Data Validation Module

This module provides comprehensive data validation functions for the customer segmentation pipeline.
Ensures data quality, integrity, and compliance with business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime


class DataValidator:
    """
    Comprehensive data validation class for customer segmentation data.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.logger = logging.getLogger(__name__)
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate raw transaction data against expected schema and business rules.
        
        Args:
            df (pd.DataFrame): Raw transaction dataframe
            
        Returns:
            Dict: Validation results with status and issues
        """
        results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                         'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['status'] = 'fail'
            results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        expected_types = {
            'InvoiceNo': 'object',
            'StockCode': 'object', 
            'Description': 'object',
            'Quantity': 'int64',
            'UnitPrice': 'float64',
            'CustomerID': 'object',
            'Country': 'object'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    results['warnings'].append(
                        f"Column {col} has type {actual_type}, expected {expected_type}"
                    )
        
        # Check for negative values
        if 'Quantity' in df.columns:
            neg_quantity = (df['Quantity'] < 0).sum()
            if neg_quantity > 0:
                results['warnings'].append(f"Found {neg_quantity} transactions with negative quantity")
        
        if 'UnitPrice' in df.columns:
            neg_price = (df['UnitPrice'] < 0).sum()
            if neg_price > 0:
                results['warnings'].append(f"Found {neg_price} transactions with negative unit price")
        
        # Check date format
        if 'InvoiceDate' in df.columns:
            try:
                pd.to_datetime(df['InvoiceDate'])
            except:
                results['status'] = 'fail'
                results['issues'].append("InvoiceDate contains invalid date format")
        
        # Calculate statistics
        results['statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
            'null_values': df.isnull().sum().to_dict()
        }
        
        self.validation_results['raw_data'] = results
        return results
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed transaction data.
        
        Args:
            df (pd.DataFrame): Processed transaction dataframe
            
        Returns:
            Dict: Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required columns after processing
        required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                         'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country', 'TotalPrice']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['status'] = 'fail'
            results['issues'].append(f"Missing processed columns: {missing_columns}")
        
        # Check for negative values (should be removed after processing)
        if 'Quantity' in df.columns:
            neg_quantity = (df['Quantity'] < 0).sum()
            if neg_quantity > 0:
                results['warnings'].append(f"Still contains {neg_quantity} negative quantity transactions")
        
        if 'TotalPrice' in df.columns:
            neg_total = (df['TotalPrice'] < 0).sum()
            if neg_total > 0:
                results['warnings'].append(f"Still contains {neg_total} negative total price transactions")
        
        # Check CustomerID format (should be numeric or NXXXX format)
        if 'CustomerID' in df.columns:
            invalid_customer_ids = df[~df['CustomerID'].astype(str).str.match(r'^\d+$|^N\d{4}$')].shape[0]
            if invalid_customer_ids > 0:
                results['warnings'].append(f"Found {invalid_customer_ids} CustomerIDs with invalid format (expected numeric or NXXXX)")
        
        # Check data quality metrics
        results['statistics'] = {
            'total_rows': len(df),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
            'date_range': {
                'min': df['InvoiceDate'].min() if 'InvoiceDate' in df.columns else None,
                'max': df['InvoiceDate'].max() if 'InvoiceDate' in df.columns else None
            },
            'new_customer_count': df['CustomerID'].astype(str).str.startswith('N').sum() if 'CustomerID' in df.columns else 0
        }
        
        self.validation_results['processed_data'] = results
        return results
    
    def validate_customer_features(self, df: pd.DataFrame) -> Dict:
        """
        Validate customer features dataframe.
        
        Args:
            df (pd.DataFrame): Customer features dataframe
            
        Returns:
            Dict: Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required RFM columns
        required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                         'TotalItems', 'UniqueProducts', 'Country', 
                         'AvgOrderValue', 'ItemsPerOrder']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['status'] = 'fail'
            results['issues'].append(f"Missing feature columns: {missing_columns}")
        
        # Validate RFM logic
        if all(col in df.columns for col in ['Recency', 'Frequency', 'Monetary']):
            # Recency should be non-negative
            if (df['Recency'] < 0).any():
                results['issues'].append("Recency contains negative values")
            
            # Frequency should be positive
            if (df['Frequency'] <= 0).any():
                results['warnings'].append("Frequency contains zero or negative values")
            
            # Monetary should be non-negative
            if (df['Monetary'] < 0).any():
                results['issues'].append("Monetary contains negative values")
        
        # Check for outliers in key metrics
        if 'Monetary' in df.columns:
            q1, q3 = df['Monetary'].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df['Monetary'] < (q1 - 1.5*iqr)) | 
                       (df['Monetary'] > (q3 + 1.5*iqr))).sum()
            if outliers > 0:
                results['warnings'].append(f"Found {outliers} monetary outliers")
        
        # Customer statistics
        results['statistics'] = {
            'total_customers': len(df),
            'avg_recency': df['Recency'].mean() if 'Recency' in df.columns else None,
            'avg_frequency': df['Frequency'].mean() if 'Frequency' in df.columns else None,
            'avg_monetary': df['Monetary'].mean() if 'Monetary' in df.columns else None,
            'total_countries': df['Country'].nunique() if 'Country' in df.columns else None,
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        self.validation_results['customer_features'] = results
        return results
    
    def validate_segments(self, df: pd.DataFrame) -> Dict:
        """
        Validate customer segments dataframe.
        
        Args:
            df (pd.DataFrame): Customer segments dataframe
            
        Returns:
            Dict: Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_columns = ['CustomerID', 'Cluster', 'Segment']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['status'] = 'fail'
            results['issues'].append(f"Missing segment columns: {missing_columns}")
        
        # Validate cluster assignments
        if 'Cluster' in df.columns:
            # Check for NaN clusters
            if df['Cluster'].isnull().any():
                results['issues'].append("Cluster assignments contain NaN values")
            
            # Check cluster distribution
            cluster_counts = df['Cluster'].value_counts()
            min_cluster_size = cluster_counts.min()
            if min_cluster_size < 10:
                results['warnings'].append(f"Smallest cluster has only {min_cluster_size} customers")
        
        # Validate segment names
        if 'Segment' in df.columns:
            # Check for NaN segments
            if df['Segment'].isnull().any():
                results['issues'].append("Segment assignments contain NaN values")
            
            # Check segment distribution
            segment_counts = df['Segment'].value_counts()
            if len(segment_counts) < 2:
                results['warnings'].append("Only one segment found - segmentation may not be meaningful")
        
        # Segment statistics
        results['statistics'] = {
            'total_customers': len(df),
            'num_clusters': df['Cluster'].nunique() if 'Cluster' in df.columns else None,
            'num_segments': df['Segment'].nunique() if 'Segment' in df.columns else None,
            'segment_distribution': df['Segment'].value_counts().to_dict() if 'Segment' in df.columns else None,
            'cluster_distribution': df['Cluster'].value_counts().to_dict() if 'Cluster' in df.columns else None
        }
        
        self.validation_results['segments'] = results
        return results
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns:
            str: Formatted validation report
        """
        report = []
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for stage, results in self.validation_results.items():
            status_icon = "✅" if results['status'] == 'pass' else "❌"
            report.append(f"{status_icon} {stage.upper().replace('_', ' ')}")
            report.append("-" * 30)
            
            if results['issues']:
                report.append("Issues:")
                for issue in results['issues']:
                    report.append(f"  ❌ {issue}")
            
            if results['warnings']:
                report.append("Warnings:")
                for warning in results['warnings']:
                    report.append(f"  ⚠️  {warning}")
            
            if not results['issues'] and not results['warnings']:
                report.append("✅ All checks passed")
            
            report.append("")
            
            # Add statistics
            if results['statistics']:
                report.append("Statistics:")
                for key, value in results['statistics'].items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            report.append(f"    {sub_key}: {sub_value}")
                    else:
                        report.append(f"  {key}: {value}")
                report.append("")
        
        return "\n".join(report)
    
    def save_validation_report(self, filepath: str) -> None:
        """
        Save validation report to file.
        
        Args:
            filepath (str): Path to save the report
        """
        report = self.generate_validation_report()
        with open(filepath, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Validation report saved to: {filepath}")


def validate_pipeline_data(raw_df: pd.DataFrame, 
                       processed_df: pd.DataFrame = None,
                       features_df: pd.DataFrame = None,
                       segments_df: pd.DataFrame = None,
                       save_report: bool = True,
                       report_path: str = None) -> Dict:
    """
    Validate all stages of the customer segmentation pipeline.
    
    Args:
        raw_df (pd.DataFrame): Raw transaction data
        processed_df (pd.DataFrame): Processed transaction data
        features_df (pd.DataFrame): Customer features data
        segments_df (pd.DataFrame): Customer segments data
        save_report (bool): Whether to save validation report
        report_path (str): Path to save report
        
    Returns:
        Dict: Complete validation results
    """
    validator = DataValidator()
    
    # Validate each stage
    validator.validate_raw_data(raw_df)
    
    if processed_df is not None:
        validator.validate_processed_data(processed_df)
    
    if features_df is not None:
        validator.validate_customer_features(features_df)
    
    if segments_df is not None:
        validator.validate_segments(segments_df)
    
    # Save report if requested
    if save_report:
        if report_path is None:
            report_path = '../reports/data_validation_report.txt'
        validator.save_validation_report(report_path)
    
    return validator.validation_results


if __name__ == "__main__":
    # Example usage
    print("Data Validation Module")
    print("=" * 30)
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'InvoiceNo': ['12345', '12346', '12347'],
        'StockCode': ['A001', 'A002', 'A003'],
        'Description': ['Product A', 'Product B', 'Product C'],
        'Quantity': [10, 5, 8],
        'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'UnitPrice': [10.99, 5.99, 12.99],
        'CustomerID': ['C001', 'C002', 'C003'],
        'Country': ['UK', 'US', 'UK']
    })
    
    validator = DataValidator()
    results = validator.validate_raw_data(sample_data)
    print(validator.generate_validation_report())
