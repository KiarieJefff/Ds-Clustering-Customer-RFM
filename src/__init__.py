"""
Customer Segmentation Package

This package provides a complete customer segmentation pipeline including:
- Data preprocessing and cleaning
- Feature engineering for RFM analysis
- K-means clustering for customer segmentation
- Visualization and reporting tools
- Configuration management

Modules:
- data_preprocessing: Data cleaning and preprocessing utilities
- feature_engineering: RFM feature creation and customer metrics
- clustering: K-means clustering and segment analysis
- visualization: EDA and results visualization
- config: Configuration constants and settings
- main: Main pipeline orchestration script
"""

__version__ = "1.0.0"
__author__ = "Customer Segmentation Team"
__email__ = "team@shoppy.com"

# Import main classes and functions for easy access
from .data_preprocessing import (
    preprocess_data,
    load_data,
    handle_missing_values,
    remove_duplicates,
    enforce_dtypes,
    remove_outliers_iqr,
    filter_viable_orders
)

from .feature_engineering import (
    create_customer_features,
    calculate_rfm_features,
    calculate_additional_features,
    scale_features,
    get_default_feature_columns,
    get_rfm_feature_columns
)

from .clustering import (
    find_optimal_clusters,
    perform_kmeans_clustering,
    create_cluster_profiles,
    define_segment_names,
    assign_segments,
    save_model,
    load_model,
    predict_new_customer
)

from .visualization import (
    set_plot_style,
    plot_outlier_detection,
    plot_distributions,
    plot_country_distribution,
    plot_sales_trend,
    plot_rfm_segments,
    plot_cluster_scatter,
    plot_segment_comparison,
    create_customer_dashboard
)

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    get_file_path,
    create_directories,
    validate_paths
)

__all__ = [
    # Data preprocessing
    'preprocess_data',
    'load_data',
    'handle_missing_values',
    'remove_duplicates',
    'enforce_dtypes',
    'remove_outliers_iqr',
    'filter_viable_orders',
    
    # Feature engineering
    'create_customer_features',
    'calculate_rfm_features',
    'calculate_additional_features',
    'scale_features',
    'get_default_feature_columns',
    'get_rfm_feature_columns',
    
    # Clustering
    'find_optimal_clusters',
    'perform_kmeans_clustering',
    'create_cluster_profiles',
    'define_segment_names',
    'assign_segments',
    'save_model',
    'load_model',
    'predict_new_customer',
    
    # Visualization
    'set_plot_style',
    'plot_outlier_detection',
    'plot_distributions',
    'plot_country_distribution',
    'plot_sales_trend',
    'plot_rfm_segments',
    'plot_cluster_scatter',
    'plot_segment_comparison',
    'create_customer_dashboard',
    
    # Configuration
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'get_file_path',
    'create_directories',
    'validate_paths'
]
