# Customer Clustering Segmentation 
### **Business Problem**

**Shoppy**, an e-commerce company, is experiencing declining customer engagement and inefficient marketing spending. They are sending blanket marketing campaigns to all customers, resulting in low response rates and poor ROI. Customers have different behaviors, needs, and values to our business, but they are treating them all the same.

This directory contains the complete source code for the customer segmentation pipeline that transforms raw retail transaction data into actionable customer segments.

## 📁 Directory Structure

```
src/
├── __init__.py              # Package initialization and imports
├── README.md                # This file
├── config.py                # Configuration constants and settings
├── data_preprocessing.py    # Data cleaning and preprocessing utilities
├── feature_engineering.py   # RFM feature creation and customer metrics
├── clustering.py           # K-means clustering and segment analysis
├── visualization.py        # EDA and results visualization
└── main.py                 # Main pipeline orchestration script
```

## 🚀 Quick Start

### Running the Complete Pipeline

```bash
# Navigate to the src directory
cd src

# Run the complete pipeline
python main.py

# Run with specific number of clusters
python main.py --n-clusters 4

# Run with EDA visualizations
python main.py --run-eda --create-visualizations

# Skip preprocessing if data is already cleaned
python main.py --skip-preprocessing --skip-features
```

### Using Individual Modules

```python
# Import the package
from src import preprocess_data, create_customer_features, perform_kmeans_clustering
import pandas as pd

# Step 1: Preprocess data
df_clean = preprocess_data('../data/raw/Online_Retail.csv')

# Step 2: Create customer features
customer_features = create_customer_features(df_clean)

# Step 3: Perform clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_features[['Recency', 'Frequency', 'Monetary']])
kmeans = perform_kmeans_clustering(X_scaled, n_clusters=3)
```

## 📋 Module Descriptions

### 1. `config.py`
Central configuration file containing:
- File paths and directory structure
- Data processing parameters
- Clustering settings
- Visualization preferences
- Business context and metadata

### 2. `data_preprocessing.py`
Handles data cleaning and preparation:
- Missing value handling
- Duplicate removal
- Data type enforcement
- Outlier detection and removal
- Data validation

**Key Functions:**
- `preprocess_data()` - Complete preprocessing pipeline
- `handle_missing_values()` - Handle missing CustomerID and Description
- `remove_outliers_iqr()` - Remove outliers using IQR method
- `filter_viable_orders()` - Remove cancelled orders

### 3. `feature_engineering.py`
Creates customer-level features for segmentation:
- RFM (Recency, Frequency, Monetary) analysis
- Additional behavioral metrics
- Feature scaling for clustering

**Key Functions:**
- `create_customer_features()` - Complete feature engineering pipeline
- `calculate_rfm_features()` - Calculate RFM metrics
- `calculate_additional_features()` - Calculate AvgOrderValue, ItemsPerOrder
- `scale_features()` - Standardize features for clustering

### 4. `clustering.py`
Implements customer segmentation using K-means:
- Optimal cluster selection (Elbow Method, Silhouette Score)
- K-means clustering
- Segment profiling and naming
- Model persistence

**Key Functions:**
- `find_optimal_clusters()` - Determine optimal number of clusters
- `perform_kmeans_clustering()` - Execute K-means clustering
- `create_cluster_profiles()` - Analyze cluster characteristics
- `define_segment_names()` - Assign meaningful segment names

### 5. `visualization.py`
Creates comprehensive visualizations:
- Exploratory data analysis plots
- Cluster analysis visualizations
- Segment comparison charts
- Customer dashboard

**Key Functions:**
- `plot_outlier_detection()` - Boxplot analysis
- `plot_rfm_segments()` - Segment RFM characteristics
- `plot_cluster_scatter()` - 2D cluster visualization
- `create_customer_dashboard()` - Complete visualization suite

### 6. `main.py`
Orchestrates the complete pipeline:
- Command-line interface
- Logging and error handling
- Report generation
- Workflow management

**Usage:**
```bash
python main.py --help
```

## 🎯 Business Context

This pipeline addresses **Shoppy's** challenge of declining customer engagement and inefficient marketing spending by:

1. **Problem**: Blanket marketing campaigns with low response rates
2. **Solution**: Customer segmentation for personalized marketing
3. **Goals**: 
   - Increase campaign response rate by 20%
   - Reduce churn rate by 15%
   - Improve marketing ROI by 25%
   - Increase customer lifetime value by 18%

## 📊 Output Files

The pipeline generates several output files:

### Data Files
- `../data/processed/Online_Retail_Cleaned.csv` - Cleaned transaction data
- `../data/processed/Customer_RFM_Features.csv` - Customer features
- `../data/processed/Customer_Segments.csv` - Final segment assignments

### Model Files
- `../models/kmeans_customer_segmentation.pkl` - Trained K-means model
- `../models/scaler_customer_segmentation.pkl` - Feature scaler

### Visualization Files
- `../data/processed/images/` - Directory containing all plots and charts

### Reports
- `../reports/segmentation_report.txt` - Summary of segmentation results

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Clustering settings
CLUSTERING = {
    'max_clusters': 10,
    'min_clusters': 2,
    'random_state': 42,
    'default_n_clusters': 3
}

# Feature selection
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
ADDITIONAL_FEATURES = ['TotalItems', 'UniqueProducts', 'AvgOrderValue']

# Business thresholds
SEGMENT_DEFINITIONS = {
    'loyal_high_spenders': {
        'frequency_threshold': 100,
        'monetary_threshold': 50000
    }
}
```

## 🧪 Testing

To test individual modules:

```python
# Test data preprocessing
python -c "from data_preprocessing import preprocess_data; print(preprocess_data('../data/raw/Online_Retail.csv').shape)"

# Test feature engineering
python -c "from feature_engineering import create_customer_features; import pandas as pd; df = pd.read_csv('../data/processed/Online_Retail_Cleaned.csv'); print(create_customer_features(df).shape)"

# Test clustering
python clustering.py
```

## 📝 Dependencies

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `joblib` - Model persistence

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 🐛 Troubleshooting

### Common Issues

1. **File not found errors**: Ensure data files exist in `../data/raw/`
2. **Memory issues**: Use smaller datasets or increase available memory
3. **Import errors**: Check Python path and module installation

### Logging

Pipeline logs are saved to `../logs/customer_segmentation.log` with configurable levels:
```bash
python main.py --log-level DEBUG
```

## 🤝 Contributing

When adding new features:
1. Follow the existing code style
2. Add comprehensive docstrings
3. Update this README
4. Test with sample data
5. Update configuration if needed

## 📞 Support

For questions or issues:
1. Check the log files for error details
2. Review the configuration settings
3. Validate input data format
4. Consult the business context documentation

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-06  
**Purpose**: Customer Segmentation for Shoppy E-commerce Platform
