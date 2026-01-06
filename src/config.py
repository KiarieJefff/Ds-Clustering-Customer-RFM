"""
Configuration Module

This module contains configuration constants and settings for the customer segmentation pipeline.
"""

import os
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
SRC_DIR = PROJECT_ROOT / "src"


# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "Online_Retail.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "Online_Retail_Cleaned.csv"
CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "Customer_RFM_Features.csv"
CUSTOMER_SEGMENTS_FILE = PROCESSED_DATA_DIR / "Customer_Segments.csv"


# Model file paths
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_customer_segmentation.pkl"
SCALER_MODEL_FILE = MODELS_DIR / "scaler_customer_segmentation.pkl"


# Report file paths
SEGMENTATION_REPORT_FILE = REPORTS_DIR / "segmentation_report.txt"
LOG_FILE = LOGS_DIR / "customer_segmentation.log"


# Data processing settings
MISSING_VALUE_HANDLING = {
    'description': 'drop',  # 'drop' or 'fill'
    'customer_id': 'fill',  # 'drop' or 'fill'
    'customer_id_fill_value': 'Guest'
}

DUPLICATE_HANDLING = {
    'keep': 'last'  # 'first', 'last', or False
}

OUTLIER_REMOVAL = {
    'method': 'iqr',  # 'iqr' or 'zscore'
    'columns': ['UnitPrice', 'Quantity'],
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3.0
}

DATA_TYPES = {
    'InvoiceNo': 'object',
    'StockCode': 'object',
    'Description': 'object',
    'Quantity': 'Int64',
    'InvoiceDate': 'datetime64[ns]',
    'UnitPrice': 'float64',
    'CustomerID': 'object',
    'Country': 'object'
}


# Feature engineering settings
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
ADDITIONAL_FEATURES = ['TotalItems', 'UniqueProducts', 'AvgOrderValue', 'ItemsPerOrder']
ALL_FEATURES = RFM_FEATURES + ADDITIONAL_FEATURES

DEFAULT_FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 
                          'AvgOrderValue', 'ItemsPerOrder']


# Clustering settings
CLUSTERING = {
    'algorithm': 'kmeans',
    'max_clusters': 10,
    'min_clusters': 2,
    'random_state': 42,
    'n_init': 10,
    'default_n_clusters': 3
}

SEGMENT_DEFINITIONS = {
    'loyal_high_spenders': {
        'frequency_threshold': 100,
        'monetary_threshold': 50000,
        'name': 'Loyal High-Spenders'
    },
    'at_risk_customers': {
        'recency_threshold': 300,
        'frequency_threshold': 5,
        'name': 'At-Risk Customers'
    },
    'regular_customers': {
        'frequency_threshold': 10,
        'monetary_threshold': 2000,
        'name': 'Regular Customers'
    },
    'recent_engaged': {
        'recency_threshold': 100,
        'frequency_threshold': 5,
        'name': 'Recent Engaged Customers'
    },
    'low_value': {
        'monetary_threshold': 500,
        'name': 'Low-Value Customers'
    },
    'bargain_hunters': {
        'name': 'Bargain Hunters'
    }
}


# Visualization settings
PLOT_STYLE = {
    'style': 'whitegrid',
    'palette': 'muted',
    'figure_size': (12, 8),
    'font_size': 12,
    'dpi': 300
}

VISUALIZATION_SETTINGS = {
    'save_plots': True,
    'show_plots': True,
    'plot_format': 'png',
    'bbox_inches': 'tight'
}


# Logging settings
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True
}


# Business context
BUSINESS_CONTEXT = {
    'company_name': 'Shoppy',
    'business_problem': 'Declining customer engagement and inefficient marketing spending',
    'objective': 'Segment customers into distinct behavioral groups for personalized marketing',
    'success_metrics': {
        'campaign_response_rate_increase': 0.20,  # 20%
        'churn_rate_reduction': 0.15,  # 15%
        'marketing_roi_improvement': 0.25,  # 25%
        'clv_increase': 0.18  # 18%
    },
    'stakeholders': [
        'Marketing Department',
        'Business Strategy Team',
        'Customer Success Team',
        'E-commerce Product Managers'
    ]
}


# Data source information
DATA_SOURCE = {
    'origin': 'UCI Machine Learning Repository',
    'dataset_id': '352',
    'url': 'https://archive.ics.uci.edu/dataset/352/online+retail',
    'time_period': 'Dec 2010 - Dec 2011',
    'business_type': 'UK-based online retailer',
    'description': 'Complete transaction history for customer behavior analysis'
}


# Model validation settings
VALIDATION = {
    'test_size': 0.2,
    'cross_validation_folds': 5,
    'silhouette_threshold': 0.5,
    'min_cluster_size': 50
}


# Performance settings
PERFORMANCE = {
    'chunk_size': 10000,  # For large datasets
    'n_jobs': -1,  # Use all available cores
    'memory_efficient': True
}


def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        IMAGES_DIR,
        SRC_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_file_path(file_type: str) -> Path:
    """
    Get file path based on file type.
    
    Args:
        file_type (str): Type of file ('raw_data', 'processed_data', 'features', 
                      'segments', 'kmeans_model', 'scaler_model', 'report', 'log')
    
    Returns:
        Path: File path
    """
    file_paths = {
        'raw_data': RAW_DATA_FILE,
        'processed_data': PROCESSED_DATA_FILE,
        'features': CUSTOMER_FEATURES_FILE,
        'segments': CUSTOMER_SEGMENTS_FILE,
        'kmeans_model': KMEANS_MODEL_FILE,
        'scaler_model': SCALER_MODEL_FILE,
        'report': SEGMENTATION_REPORT_FILE,
        'log': LOG_FILE
    }
    
    return file_paths.get(file_type, None)


def validate_paths():
    """Validate that all required directories exist or can be created."""
    try:
        create_directories()
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("Configuration Settings:")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data File: {RAW_DATA_FILE}")
    print(f"Processed Data File: {PROCESSED_DATA_FILE}")
    print(f"Customer Features File: {CUSTOMER_FEATURES_FILE}")
    print(f"Customer Segments File: {CUSTOMER_SEGMENTS_FILE}")
    print(f"KMeans Model File: {KMEANS_MODEL_FILE}")
    print(f"Scaler Model File: {SCALER_MODEL_FILE}")
    print(f"Report File: {SEGMENTATION_REPORT_FILE}")
    print(f"Log File: {LOG_FILE}")
    
    # Validate paths
    if validate_paths():
        print("\nAll directories validated successfully!")
    else:
        print("\nError validating directories!")
