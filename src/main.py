"""
Main Pipeline Script

This script orchestrates the complete customer segmentation pipeline including:
1. Data preprocessing
2. Feature engineering
3. Clustering analysis
4. Visualization and reporting
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import preprocess_data, save_processed_data
from feature_engineering import create_customer_features, save_customer_features, get_rfm_feature_columns
from clustering import (find_optimal_clusters, plot_elbow_method, plot_silhouette_scores,
                       perform_kmeans_clustering, create_cluster_profiles, 
                       define_segment_names, assign_segments, save_model)
from visualization import (set_plot_style, plot_outlier_detection, plot_distributions,
                          plot_country_distribution, plot_sales_trend, plot_rfm_segments,
                          plot_cluster_scatter, create_customer_dashboard)


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/customer_segmentation.log'),
            logging.StreamHandler()
        ]
    )


def create_directories() -> None:
    """Create necessary directories for the pipeline."""
    directories = [
        '../data/processed',
        '../data/processed/images',
        '../models',
        '../reports',
        '../logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run exploratory data analysis and create visualizations.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        output_dir (str): Directory to save plots
    """
    logging.info("Running Exploratory Data Analysis...")
    
    # Set plot style
    set_plot_style()
    
    # Outlier detection
    plot_outlier_detection(df, ['UnitPrice', 'Quantity'], 
                          save_path=os.path.join(output_dir, 'outliers.png'))
    
    # Distributions
    plot_distributions(df, ['UnitPrice', 'Quantity'],
                       save_path=os.path.join(output_dir, 'distributions.png'))
    
    # Country distribution
    plot_country_distribution(df, save_path=os.path.join(output_dir, 'country_distribution.png'))
    
    # Sales trend (add Sales column first)
    df['Sales'] = df['UnitPrice'] * df['Quantity']
    plot_sales_trend(df, start_date='2010-04-01',
                    save_path=os.path.join(output_dir, 'sales_trend.png'))
    
    logging.info("EDA completed and plots saved.")


def run_clustering_pipeline(df_features: pd.DataFrame, 
                          n_clusters: int = None,
                          save_models: bool = True) -> pd.DataFrame:
    """
    Run the complete clustering pipeline.
    
    Args:
        df_features (pd.DataFrame): Customer features dataframe
        n_clusters (int): Number of clusters (if None, will find optimal)
        save_models (bool): Whether to save trained models
        
    Returns:
        pd.DataFrame: Dataframe with cluster and segment assignments
    """
    logging.info("Starting clustering pipeline...")
    
    # Select RFM features for clustering
    feature_cols = get_rfm_feature_columns()
    X = df_features[feature_cols].copy()
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters if not specified
    if n_clusters is None:
        logging.info("Finding optimal number of clusters...")
        cluster_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled)
        
        # Plot elbow method and silhouette scores
        plot_elbow_method(cluster_range, wcss, 
                         save_path='../data/processed/images/elbow_method.png')
        plot_silhouette_scores(cluster_range, silhouette_scores,
                              save_path='../data/processed/images/silhouette_scores.png')
        
        # Choose optimal k (based on highest silhouette score)
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        logging.info(f"Optimal number of clusters: {optimal_k}")
        n_clusters = optimal_k
    
    # Perform clustering
    logging.info(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = perform_kmeans_clustering(X_scaled, n_clusters)
    
    # Add cluster assignments to dataframe
    df_features['Cluster'] = kmeans.labels_
    
    # Create cluster profiles
    cluster_profile = create_cluster_profiles(df_features)
    logging.info("Cluster profiles created:")
    logging.info(cluster_profile.to_string())
    
    # Define segment names
    segment_names = define_segment_names(cluster_profile)
    
    # Assign segments
    df_features = assign_segments(df_features, segment_names)
    
    # Save models if requested
    if save_models:
        save_model(kmeans, scaler, 
                  '../models/kmeans_customer_segmentation.pkl',
                  '../models/scaler_customer_segmentation.pkl')
    
    # Save results
    df_features.to_csv('../data/processed/Customer_Segments.csv', index=False)
    
    logging.info("Clustering pipeline completed successfully.")
    
    return df_features


def generate_report(df_segments: pd.DataFrame) -> None:
    """
    Generate a summary report of the segmentation results.
    
    Args:
        df_segments (pd.DataFrame): Dataframe with segment assignments
    """
    logging.info("Generating segmentation report...")
    
    # Segment summary
    segment_summary = df_segments.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'AvgOrderValue': 'mean'
    }).round(2)
    
    segment_summary.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 
                               'Avg_Monetary', 'Total_Monetary', 'Avg_Order_Value']
    
    # Save report
    with open('../reports/segmentation_report.txt', 'w') as f:
        f.write("CUSTOMER SEGMENTATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SEGMENT SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(segment_summary.to_string())
        f.write("\n\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        
        # Calculate some insights
        total_customers = len(df_segments)
        total_revenue = df_segments['Monetary'].sum()
        
        f.write(f"Total Customers: {total_customers:,}\n")
        f.write(f"Total Revenue: ${total_revenue:,.2f}\n")
        f.write(f"Average Revenue per Customer: ${total_revenue/total_customers:.2f}\n\n")
        
        # Best performing segment
        best_segment = segment_summary.loc[segment_summary['Avg_Monetary'].idxmax()]
        f.write(f"Highest Value Segment: {best_segment.name}\n")
        f.write(f"  - Average Monetary Value: ${best_segment['Avg_Monetary']:,.2f}\n")
        f.write(f"  - Customer Count: {best_segment['Count']:,}\n\n")
        
        # Largest segment
        largest_segment = segment_summary.loc[segment_summary['Count'].idxmax()]
        f.write(f"Largest Segment: {largest_segment.name}\n")
        f.write(f"  - Customer Count: {largest_segment['Count']:,}\n")
        f.write(f"  - Average Monetary Value: ${largest_segment['Avg_Monetary']:,.2f}\n")
    
    logging.info("Report saved to ../reports/segmentation_report.txt")


def main(args):
    """Main pipeline function."""
    # Setup
    setup_logging(args.log_level)
    create_directories()
    
    logging.info("Starting Customer Segmentation Pipeline...")
    
    try:
        # Step 1: Data Preprocessing
        logging.info("Step 1: Data Preprocessing")
        if args.skip_preprocessing and os.path.exists('../data/processed/Online_Retail_Cleaned.csv'):
            logging.info("Loading preprocessed data...")
            df_processed = pd.read_csv('../data/processed/Online_Retail_Cleaned.csv')
        else:
            df_processed = preprocess_data('../data/raw/Online_Retail.csv')
            save_processed_data(df_processed, '../data/processed/Online_Retail_Cleaned.csv')
        
        logging.info(f"Data preprocessing completed. Shape: {df_processed.shape}")
        
        # Step 2: Exploratory Data Analysis
        if args.run_eda:
            logging.info("Step 2: Exploratory Data Analysis")
            run_eda(df_processed, '../data/processed/images/')
        
        # Step 3: Feature Engineering
        logging.info("Step 3: Feature Engineering")
        if args.skip_features and os.path.exists('../data/processed/Customer_RFM_Features.csv'):
            logging.info("Loading pre-computed features...")
            df_features = pd.read_csv('../data/processed/Customer_RFM_Features.csv')
        else:
            df_features = create_customer_features(df_processed)
            save_customer_features(df_features, '../data/processed/Customer_RFM_Features.csv')
        
        logging.info(f"Feature engineering completed. Shape: {df_features.shape}")
        
        # Step 4: Clustering
        logging.info("Step 4: Customer Clustering")
        df_segments = run_clustering_pipeline(df_features, n_clusters=args.n_clusters)
        
        # Step 5: Visualization
        if args.create_visualizations:
            logging.info("Step 5: Creating Visualizations")
            set_plot_style()
            create_customer_dashboard(df_segments)
        
        # Step 6: Report Generation
        logging.info("Step 6: Report Generation")
        generate_report(df_segments)
        
        logging.info("Pipeline completed successfully!")
        logging.info(f"Results saved to ../data/processed/Customer_Segments.csv")
        logging.info(f"Report saved to ../reports/segmentation_report.txt")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Segmentation Pipeline")
    
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters for K-means (if None, will find optimal)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing if cleaned data exists')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering if features exist')
    parser.add_argument('--run-eda', action='store_true',
                       help='Run exploratory data analysis')
    parser.add_argument('--create-visualizations', action='store_true', default=True,
                       help='Create visualization dashboard')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    main(args)
