"""
Model Evaluation Module

This module provides comprehensive evaluation metrics for customer segmentation models.
Includes clustering validation, business metrics, and segment quality assessment.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation for customer segmentation.
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Calculate comprehensive clustering metrics.
        
        Args:
            X (np.ndarray): Feature matrix
            labels (np.ndarray): Cluster labels
            
        Returns:
            Dict: Clustering metrics
        """
        n_clusters = len(np.unique(labels))
        
        # Basic metrics
        metrics = {
            'n_clusters': n_clusters,
            'n_samples': len(labels),
            'cluster_sizes': [np.sum(labels == i) for i in range(n_clusters)]
        }
        
        # Internal validation metrics
        if n_clusters > 1 and n_clusters < len(X):
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            except:
                metrics['silhouette_score'] = -1
                metrics['davies_bouldin_score'] = -1
                metrics['calinski_harabasz_score'] = -1
        else:
            metrics['silhouette_score'] = -1
            metrics['davies_bouldin_score'] = -1
            metrics['calinski_harabasz_score'] = -1
        
        # Cluster balance metrics
        cluster_sizes = np.array(metrics['cluster_sizes'])
        if len(cluster_sizes) > 0:
            metrics['min_cluster_size'] = np.min(cluster_sizes)
            metrics['max_cluster_size'] = np.max(cluster_sizes)
            metrics['mean_cluster_size'] = np.mean(cluster_sizes)
            metrics['std_cluster_size'] = np.std(cluster_sizes)
            metrics['balance_ratio'] = metrics['min_cluster_size'] / metrics['max_cluster_size']
        
        return metrics
    
    def business_metrics(self, df: pd.DataFrame, 
                      cluster_col: str = 'Cluster',
                      segment_col: str = 'Segment',
                      monetary_col: str = 'Monetary',
                      frequency_col: str = 'Frequency',
                      recency_col: str = 'Recency') -> Dict:
        """
        Calculate business-relevant metrics for segments.
        
        Args:
            df (pd.DataFrame): Dataframe with cluster assignments
            cluster_col (str): Cluster column name
            segment_col (str): Segment column name
            monetary_col (str): Monetary column name
            frequency_col (str): Frequency column name
            recency_col (str): Recency column name
            
        Returns:
            Dict: Business metrics
        """
        metrics = {}
        
        # Overall metrics
        total_customers = len(df)
        total_revenue = df[monetary_col].sum()
        
        metrics['total_customers'] = total_customers
        metrics['total_revenue'] = total_revenue
        metrics['avg_customer_value'] = total_revenue / total_customers
        
        # Segment-level metrics
        if segment_col in df.columns:
            segment_stats = df.groupby(segment_col).agg({
                monetary_col: ['sum', 'mean', 'count'],
                frequency_col: 'mean',
                recency_col: 'mean'
            }).round(2)
            
            segment_stats.columns = ['Total_Revenue', 'Avg_Monetary', 'Customer_Count', 
                                 'Avg_Frequency', 'Avg_Recency']
            
            # Revenue concentration
            revenue_by_segment = segment_stats['Total_Revenue']
            metrics['revenue_concentration'] = (revenue_by_segment / revenue_by_segment.sum()).to_dict()
            
            # Customer concentration
            customer_by_segment = segment_stats['Customer_Count']
            metrics['customer_concentration'] = (customer_by_segment / customer_by_segment.sum()).to_dict()
            
            # High-value segment identification
            metrics['highest_value_segment'] = revenue_by_segment.idxmax()
            metrics['largest_segment'] = customer_by_segment.idxmax()
            
            # Segment quality scores
            segment_quality = {}
            for segment in segment_stats.index:
                segment_data = df[df[segment_col] == segment]
                
                # RFM scores (normalized)
                r_score = 1 - (segment_data[recency_col].mean() / df[recency_col].max())
                f_score = segment_data[frequency_col].mean() / df[frequency_col].max()
                m_score = segment_data[monetary_col].mean() / df[monetary_col].max()
                
                segment_quality[segment] = (r_score + f_score + m_score) / 3
            
            metrics['segment_quality_scores'] = segment_quality
        
        # Cluster-level metrics (if no segments)
        elif cluster_col in df.columns:
            cluster_stats = df.groupby(cluster_col).agg({
                monetary_col: ['sum', 'mean', 'count'],
                frequency_col: 'mean',
                recency_col: 'mean'
            }).round(2)
            
            cluster_stats.columns = ['Total_Revenue', 'Avg_Monetary', 'Customer_Count', 
                                 'Avg_Frequency', 'Avg_Recency']
            
            metrics['cluster_stats'] = cluster_stats.to_dict()
        
        return metrics
    
    def segment_stability(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        customer_col: str = 'CustomerID',
                        cluster_col: str = 'Cluster') -> Dict:
        """
        Evaluate segment stability over time.
        
        Args:
            df1 (pd.DataFrame): First time period data
            df2 (pd.DataFrame): Second time period data
            customer_col (str): Customer ID column
            cluster_col (str): Cluster column
            
        Returns:
            Dict: Stability metrics
        """
        # Merge dataframes
        merged = df1[[customer_col, cluster_col]].merge(
            df2[[customer_col, cluster_col]], 
            on=customer_col, 
            suffixes=('_t1', '_t2')
        )
        
        # Calculate stability metrics
        total_customers = len(merged)
        stable_customers = (merged[f'{cluster_col}_t1'] == merged[f'{cluster_col}_t2']).sum()
        
        stability_rate = stable_customers / total_customers if total_customers > 0 else 0
        
        # Cluster transition matrix
        transition_matrix = pd.crosstab(
            merged[f'{cluster_col}_t1'], 
            merged[f'{cluster_col}_t2'],
            normalize='index'
        )
        
        metrics = {
            'stability_rate': stability_rate,
            'total_customers_compared': total_customers,
            'stable_customers': stable_customers,
            'transition_matrix': transition_matrix.to_dict()
        }
        
        # Calculate ARI if ground truth available
        if len(merged[f'{cluster_col}_t1'].unique()) > 1:
            metrics['adjusted_rand_index'] = adjusted_rand_score(
                merged[f'{cluster_col}_t1'], 
                merged[f'{cluster_col}_t2']
            )
        
        return metrics
    
    def segment_purity(self, df: pd.DataFrame, 
                     cluster_col: str = 'Cluster',
                     true_label_col: str = 'Segment') -> Dict:
        """
        Calculate segment purity if true labels available.
        
        Args:
            df (pd.DataFrame): Dataframe with clusters and true labels
            cluster_col (str): Cluster column
            true_label_col (str): True label column
            
        Returns:
            Dict: Purity metrics
        """
        if true_label_col not in df.columns:
            return {'error': 'True labels not available'}
        
        # Calculate purity for each cluster
        clusters = df[cluster_col].unique()
        purity_scores = {}
        
        for cluster in clusters:
            cluster_data = df[df[cluster_col] == cluster]
            if len(cluster_data) > 0:
                # Find most common true label in cluster
                label_counts = cluster_data[true_label_col].value_counts()
                most_common_count = label_counts.iloc[0]
                cluster_purity = most_common_count / len(cluster_data)
                purity_scores[cluster] = cluster_purity
        
        # Overall purity
        overall_purity = np.mean(list(purity_scores.values()))
        
        return {
            'cluster_purity': purity_scores,
            'overall_purity': overall_purity,
            'clusters_evaluated': len(clusters)
        }
    
    def segment_lift_analysis(self, df: pd.DataFrame,
                          cluster_col: str = 'Cluster',
                          monetary_col: str = 'Monetary',
                          frequency_col: str = 'Frequency') -> Dict:
        """
        Calculate lift analysis for segments.
        
        Args:
            df (pd.DataFrame): Dataframe with cluster assignments
            cluster_col (str): Cluster column
            monetary_col (str): Monetary column
            frequency_col (str): Frequency column
            
        Returns:
            Dict: Lift analysis results
        """
        # Overall averages
        overall_avg_monetary = df[monetary_col].mean()
        overall_avg_frequency = df[frequency_col].mean()
        
        lift_results = {}
        
        for cluster in df[cluster_col].unique():
            cluster_data = df[df[cluster_col] == cluster]
            
            # Cluster averages
            cluster_avg_monetary = cluster_data[monetary_col].mean()
            cluster_avg_frequency = cluster_data[frequency_col].mean()
            
            # Calculate lift
            monetary_lift = cluster_avg_monetary / overall_avg_monetary
            frequency_lift = cluster_avg_frequency / overall_avg_frequency
            
            lift_results[cluster] = {
                'size': len(cluster_data),
                'avg_monetary': cluster_avg_monetary,
                'avg_frequency': cluster_avg_frequency,
                'monetary_lift': monetary_lift,
                'frequency_lift': frequency_lift,
                'combined_lift': (monetary_lift + frequency_lift) / 2
            }
        
        return lift_results
    
    def evaluate_segmentation_quality(self, df: pd.DataFrame,
                                 cluster_col: str = 'Cluster',
                                 segment_col: str = 'Segment') -> Dict:
        """
        Comprehensive segmentation quality evaluation.
        
        Args:
            df (pd.DataFrame): Dataframe with segment assignments
            cluster_col (str): Cluster column
            segment_col (str): Segment column
            
        Returns:
            Dict: Quality evaluation results
        """
        quality_scores = {}
        
        # Segment separation (using RFM features)
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        available_rfm = [col for col in rfm_cols if col in df.columns]
        
        if len(available_rfm) >= 2:
            X = df[available_rfm].values
            labels = df[cluster_col].values
            
            # Calculate separation metrics
            if len(np.unique(labels)) > 1:
                quality_scores['silhouette_separation'] = silhouette_score(X, labels)
                quality_scores['davies_bouldin_separation'] = davies_bouldin_score(X, labels)
        
        # Segment size balance
        cluster_sizes = df[cluster_col].value_counts()
        size_balance = cluster_sizes.std() / cluster_sizes.mean()
        quality_scores['size_balance'] = 1 / (1 + size_balance)  # Higher is better
        
        # Business relevance
        if segment_col in df.columns:
            # Revenue variance across segments
            revenue_by_segment = df.groupby(segment_col)['Monetary'].sum()
            revenue_cv = revenue_by_segment.std() / revenue_by_segment.mean()
            quality_scores['revenue_differentiation'] = min(revenue_cv, 1.0)  # Cap at 1
        
        # Overall quality score
        valid_scores = [v for v in quality_scores.values() if isinstance(v, (int, float)) and v >= 0]
        if valid_scores:
            quality_scores['overall_quality'] = np.mean(valid_scores)
        
        return quality_scores
    
    def generate_evaluation_report(self, df: pd.DataFrame,
                               X: np.ndarray = None,
                               cluster_col: str = 'Cluster',
                               segment_col: str = 'Segment') -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            df (pd.DataFrame): Dataframe with segment assignments
            X (np.ndarray): Feature matrix for clustering metrics
            cluster_col (str): Cluster column
            segment_col (str): Segment column
            
        Returns:
            str: Formatted evaluation report
        """
        report = []
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        # Clustering metrics
        if X is not None and cluster_col in df.columns:
            clustering_metrics = self.clustering_metrics(X, df[cluster_col].values)
            report.append("CLUSTERING METRICS")
            report.append("-" * 20)
            for key, value in clustering_metrics.items():
                if isinstance(value, list):
                    report.append(f"{key}: {value}")
                else:
                    report.append(f"{key}: {value:.4f}")
            report.append("")
        
        # Business metrics
        business_metrics = self.business_metrics(df, cluster_col, segment_col)
        report.append("BUSINESS METRICS")
        report.append("-" * 20)
        
        for key, value in business_metrics.items():
            if isinstance(value, dict):
                report.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        report.append(f"  {sub_key}: {sub_value:.2f}")
                    else:
                        report.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                report.append(f"{key}: {value:.2f}")
            else:
                report.append(f"{key}: {value}")
        report.append("")
        
        # Quality evaluation
        quality_metrics = self.evaluate_segmentation_quality(df, cluster_col, segment_col)
        report.append("SEGMENTATION QUALITY")
        report.append("-" * 20)
        for key, value in quality_metrics.items():
            if isinstance(value, float):
                report.append(f"{key}: {value:.4f}")
            else:
                report.append(f"{key}: {value}")
        
        return "\n".join(report)
    
    def plot_evaluation_summary(self, df: pd.DataFrame,
                            cluster_col: str = 'Cluster',
                            segment_col: str = 'Segment',
                            save_path: str = None) -> None:
        """
        Plot evaluation summary visualizations.
        
        Args:
            df (pd.DataFrame): Dataframe with segment assignments
            cluster_col (str): Cluster column
            segment_col (str): Segment column
            save_path (str): Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cluster size distribution
        cluster_sizes = df[cluster_col].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # Revenue by cluster
        if 'Monetary' in df.columns:
            revenue_by_cluster = df.groupby(cluster_col)['Monetary'].sum()
            axes[0, 1].bar(revenue_by_cluster.index, revenue_by_cluster.values)
            axes[0, 1].set_title('Revenue by Cluster')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Total Revenue')
        
        # Segment distribution (if available)
        if segment_col in df.columns:
            segment_sizes = df[segment_col].value_counts()
            axes[1, 0].pie(segment_sizes.values, labels=segment_sizes.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Customer Segment Distribution')
        
        # RFM characteristics by cluster
        if all(col in df.columns for col in ['Recency', 'Frequency', 'Monetary']):
            rfm_by_cluster = df.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
            
            # Normalize for comparison
            rfm_normalized = rfm_by_cluster / rfm_by_cluster.max()
            
            x = np.arange(len(rfm_normalized))
            width = 0.25
            
            axes[1, 1].bar(x - width, rfm_normalized['Recency'], width, label='Recency')
            axes[1, 1].bar(x, rfm_normalized['Frequency'], width, label='Frequency')
            axes[1, 1].bar(x + width, rfm_normalized['Monetary'], width, label='Monetary')
            
            axes[1, 1].set_title('Normalized RFM by Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Normalized Value')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(rfm_normalized.index)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def comprehensive_evaluation(df: pd.DataFrame, X: np.ndarray = None,
                         cluster_col: str = 'Cluster',
                         segment_col: str = 'Segment',
                         save_report: bool = True,
                         report_path: str = None) -> Dict:
    """
    Perform comprehensive model evaluation.
    
    Args:
        df (pd.DataFrame): Dataframe with segment assignments
        X (np.ndarray): Feature matrix
        cluster_col (str): Cluster column
        segment_col (str): Segment column
        save_report (bool): Whether to save report
        report_path (str): Path to save report
        
    Returns:
        Dict: Complete evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Generate report
    report = evaluator.generate_evaluation_report(df, X, cluster_col, segment_col)
    
    # Save report if requested
    if save_report:
        if report_path is None:
            report_path = '../reports/model_evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(report)
    
    # Calculate all metrics
    results = {
        'clustering_metrics': evaluator.clustering_metrics(X, df[cluster_col].values) if X is not None else {},
        'business_metrics': evaluator.business_metrics(df, cluster_col, segment_col),
        'quality_metrics': evaluator.evaluate_segmentation_quality(df, cluster_col, segment_col),
        'report': report
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("=" * 30)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'CustomerID': [f'C{i:03d}' for i in range(100)],
        'Cluster': np.random.choice([0, 1, 2], 100),
        'Segment': np.random.choice(['A', 'B', 'C'], 100),
        'Recency': np.random.normal(100, 30, 100),
        'Frequency': np.random.exponential(3, 100),
        'Monetary': np.random.lognormal(4, 0.5, 100)
    })
    
    # Ensure positive values
    for col in ['Recency', 'Frequency', 'Monetary']:
        sample_data[col] = np.abs(sample_data[col])
    
    # Evaluate
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(sample_data)
    print(report)
