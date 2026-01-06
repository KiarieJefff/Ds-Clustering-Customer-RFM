"""
Visualization Module

This module contains functions for creating visualizations for exploratory data analysis
and customer segmentation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import os


def set_plot_style(style: str = 'whitegrid', palette: str = 'muted') -> None:
    """
    Set the default plot style and color palette.
    
    Args:
        style (str): Seaborn style
        palette (str): Color palette
    """
    sns.set_style(style)
    sns.set_palette(palette)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_outlier_detection(df: pd.DataFrame, 
                          columns: List[str],
                          save_path: Optional[str] = None) -> None:
    """
    Plot boxplots to visualize outliers in numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): List of columns to plot
        save_path (Optional[str]): Path to save the plot
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'{col} Outliers')
        axes[i].set_ylabel(col)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_distributions(df: pd.DataFrame, 
                       columns: List[str],
                       save_path: Optional[str] = None) -> None:
    """
    Plot distribution histograms for numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): List of columns to plot
        save_path (Optional[str]): Path to save the plot
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_country_distribution(df: pd.DataFrame,
                            country_col: str = 'Country',
                            save_path: Optional[str] = None) -> None:
    """
    Plot country distribution as pie chart (UK vs Others) and bar chart (Top 10 others).
    
    Args:
        df (pd.DataFrame): Input dataframe
        country_col (str): Name of country column
        save_path (Optional[str]): Path to save the plot
    """
    # Count countries
    country_counts = df[country_col].value_counts()
    uk_count = country_counts.get('United Kingdom', 0)
    others_total = country_counts.sum() - uk_count
    
    # Prepare data for plots
    chart1_data = [uk_count, others_total]
    chart1_labels = ['United Kingdom', 'Other Countries']
    
    # Top 10 other countries
    other_countries_counts = country_counts.drop('United Kingdom', errors='ignore')
    top_10_others = other_countries_counts.head(10)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    axes[0].pie(chart1_data, labels=chart1_labels, autopct='%1.1f%%', 
                startangle=140, colors=['skyblue', 'lightgrey'])
    axes[0].set_title('UK vs. Other Countries')
    axes[0].axis('equal')
    
    # Bar chart
    top_10_others.sort_values(ascending=True).plot(kind='barh', ax=axes[1], 
                                                   color=sns.color_palette("viridis", 10))
    axes[1].set_title('Top 10 "Other" Countries (Count)')
    axes[1].set_xlabel('Count of Orders')
    axes[1].set_ylabel('Country')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sales_trend(df: pd.DataFrame,
                    date_col: str = 'InvoiceDate',
                    sales_col: str = 'Sales',
                    start_date: Optional[str] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Plot sales trend over time.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Name of date column
        sales_col (str): Name of sales column
        start_date (Optional[str]): Start date for filtering
        save_path (Optional[str]): Path to save the plot
    """
    # Prepare data
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    
    # Filter by start date if provided
    if start_date:
        df_plot = df_plot[df_plot[date_col] >= start_date]
    
    # Aggregate by month
    monthly_sales = df_plot.set_index(date_col).resample('ME')[sales_col].sum().reset_index()
    
    # Plot
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=monthly_sales, x=date_col, y=sales_col, color='royalblue', linewidth=2)
    
    plt.title('Monthly Sales Trend', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rfm_segments(df: pd.DataFrame,
                     segment_col: str = 'Segment',
                     rfm_cols: List[str] = ['Recency', 'Frequency', 'Monetary'],
                     save_path: Optional[str] = None) -> None:
    """
    Plot RFM characteristics by customer segment.
    
    Args:
        df (pd.DataFrame): Input dataframe with segments
        segment_col (str): Name of segment column
        rfm_cols (List[str]): List of RFM columns to plot
        save_path (Optional[str]): Path to save the plot
    """
    # Calculate means for each segment
    segment_means = df.groupby(segment_col)[rfm_cols].mean().reset_index()
    
    # Create subplots
    fig, axes = plt.subplots(1, len(rfm_cols), figsize=(6*len(rfm_cols), 5))
    
    titles = ['Mean Recency (Days)', 'Mean Frequency (Orders)', 'Mean Monetary (Value)']
    
    for i, (metric, title) in enumerate(zip(rfm_cols, titles)):
        sns.barplot(data=segment_means, x=segment_col, y=metric, 
                   ax=axes[i], hue=segment_col, legend=False, palette='muted')
        axes[i].set_title(title)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cluster_scatter(df: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        cluster_col: str = 'Cluster',
                        segment_col: str = 'Segment',
                        save_path: Optional[str] = None) -> None:
    """
    Plot scatter plot of clusters colored by segment.
    
    Args:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column for x-axis
        y_col (str): Column for y-axis
        cluster_col (str): Name of cluster column
        segment_col (str): Name of segment column
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(df[x_col], df[y_col], 
                         c=df[cluster_col], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=50)
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    
    # Add labels and title
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'Customer Segments: {x_col} vs {y_col}', fontsize=14)
    
    # Add legend for segments
    segments = df[segment_col].unique()
    for i, segment in enumerate(segments):
        segment_data = df[df[segment_col] == segment]
        plt.scatter([], [], c='C'+str(i), label=segment, alpha=0.8)
    
    plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_segment_comparison(df: pd.DataFrame,
                          segment_col: str = 'Segment',
                          metrics: List[str] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot comparison of different metrics across segments.
    
    Args:
        df (pd.DataFrame): Input dataframe
        segment_col (str): Name of segment column
        metrics (List[str]): List of metrics to compare
        save_path (Optional[str]): Path to save the plot
    """
    if metrics is None:
        metrics = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']
    
    # Calculate means for each segment
    segment_stats = df.groupby(segment_col)[metrics].mean()
    
    # Ensure all data is numeric and convert any object columns to float
    segment_stats = segment_stats.astype(float)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(segment_stats.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Mean Value'})
    
    plt.title('Segment Characteristics Heatmap', fontsize=14)
    plt.xlabel('Customer Segments', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_customer_dashboard(df: pd.DataFrame,
                            output_dir: str = "../data/processed/images/") -> None:
    """
    Create a comprehensive customer segmentation dashboard.
    
    Args:
        df (pd.DataFrame): Input dataframe with customer segments
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    set_plot_style()
    
    # 1. RFM Segments
    plot_rfm_segments(df, save_path=os.path.join(output_dir, 'rfm_segments.png'))
    
    # 2. Segment Comparison Heatmap
    plot_segment_comparison(df, save_path=os.path.join(output_dir, 'segment_heatmap.png'))
    
    # 3. Cluster Scatter Plots
    plot_cluster_scatter(df, 'Frequency', 'Monetary', 
                        save_path=os.path.join(output_dir, 'frequency_monetary_scatter.png'))
    
    plot_cluster_scatter(df, 'Recency', 'Monetary', 
                        save_path=os.path.join(output_dir, 'recency_monetary_scatter.png'))
    
    print(f"Dashboard plots saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    # Load customer segments data
    df = pd.read_csv("../data/processed/Customer_Segments.csv")
    
    # Create output directory
    output_dir = "../data/processed/images/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    set_plot_style()
    
    # Create visualizations
    plot_rfm_segments(df, save_path=os.path.join(output_dir, 'rfm_segments.png'))
    plot_segment_comparison(df, save_path=os.path.join(output_dir, 'segment_heatmap.png'))
    plot_cluster_scatter(df, 'Frequency', 'Monetary', 
                        save_path=os.path.join(output_dir, 'frequency_monetary_scatter.png'))
    
    # Create full dashboard
    create_customer_dashboard(df)
    
    print("Visualization examples completed!")
