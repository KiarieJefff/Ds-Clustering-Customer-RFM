"""
Data Preprocessing Module

This module contains functions for cleaning and preprocessing the retail transaction data
for customer segmentation analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw retail data from CSV file.
    
    Args:
        file_path (str): Path to the raw data file
        
    Returns:
        pd.DataFrame: Raw transaction data
    """
    return pd.read_csv(file_path)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and summarize missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Summary of missing values
    """
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isna().sum(),
        'Missing_Percent': df.isna().mean() * 100
    }).query("Missing_Count > 0").round(2)
    
    return missing_summary


def generate_new_customer_id() -> str:
    """
    Generate a new customer ID in format NXXXX where XXXX is random from 1001-9999.
    
    Returns:
        str: New customer ID
    """
    random_number = np.random.randint(1001, 10000)
    return f"N{random_number}"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df = df.copy()
    
    # Drop rows with missing Description
    df = df.dropna(subset=['Description'])
    
    # Replace missing CustomerID with new customer IDs (NXXXX format)
    null_customer_mask = df['CustomerID'].isna()
    null_count = null_customer_mask.sum()
    
    if null_count > 0:
        # Generate unique new customer IDs for each null CustomerID
        new_ids = [generate_new_customer_id() for _ in range(null_count)]
        df.loc[null_customer_mask, 'CustomerID'] = new_ids
        
        print(f"Generated {null_count} new customer IDs (N1001-N9999 format)")
    
    return df


def remove_duplicates(df: pd.DataFrame, keep: str = 'last') -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        keep (str): Which duplicate to keep ('first', 'last', False)
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    return df.drop_duplicates(keep=keep)


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce proper data types for each column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with correct data types
    """
    df = df.copy()
    
    df['InvoiceNo'] = df['InvoiceNo'].astype('object')
    df['StockCode'] = df['StockCode'].astype('object')
    df['Description'] = df['Description'].astype('object')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').astype('Int64')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').dt.normalize()
    df['InvoiceDate'] = df['InvoiceDate'].fillna(pd.Timestamp('2010-01-12'))
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['CustomerID'] = df['CustomerID'].astype('object')
    df['Country'] = df['Country'].astype('object')
    
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to remove outliers from
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def filter_viable_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out cancelled orders (negative quantities).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with only viable orders
    """
    return df[df['Quantity'] > 0]


def preprocess_data(file_path: str, 
                   remove_outliers: bool = True,
                   outlier_columns: list = ['UnitPrice', 'Quantity']) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for retail data.
    
    Args:
        file_path (str): Path to the raw data file
        remove_outliers (bool): Whether to remove outliers
        outlier_columns (list): Columns to remove outliers from
        
    Returns:
        pd.DataFrame: Fully preprocessed data
    """
    # Load data
    df = load_data(file_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Enforce data types
    df = enforce_dtypes(df)
    
    # Remove outliers if specified
    if remove_outliers:
        df = remove_outliers_iqr(df, outlier_columns)
    
    # Filter viable orders
    df = filter_viable_orders(df)
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the processed data
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    input_path = "../data/raw/Online_Retail.csv"
    output_path = "../data/processed/Online_Retail_Cleaned.csv"
    
    # Preprocess the data
    processed_data = preprocess_data(input_path)
    
    # Save processed data
    save_processed_data(processed_data, output_path)
    
    print(f"Data preprocessing completed. Shape: {processed_data.shape}")
    print(f"Processed data saved to: {output_path}")
