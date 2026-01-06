"""
Feature Engineering Module

This module contains functions for creating RFM (Recency, Frequency, Monetary) features
and additional customer behavioral features for segmentation analysis.
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


def calculate_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total price for each transaction.
    
    Args:
        df (pd.DataFrame): Input dataframe with UnitPrice and Quantity columns
        
    Returns:
        pd.DataFrame: Dataframe with TotalPrice column added
    """
    df = df.copy()
    df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
    return df


def prepare_date_column(df: pd.DataFrame, date_column: str = 'InvoiceDate') -> pd.DataFrame:
    """
    Prepare and clean date column for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned date column
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce').dt.normalize()
    df[date_column] = df[date_column].fillna(pd.Timestamp('2010-01-12'))
    return df


def calculate_rfm_features(df: pd.DataFrame, 
                          customer_id_col: str = 'CustomerID',
                          date_col: str = 'InvoiceDate',
                          invoice_col: str = 'InvoiceNo',
                          price_col: str = 'TotalPrice',
                          quantity_col: str = 'Quantity',
                          product_col: str = 'StockCode',
                          country_col: str = 'Country') -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) features for each customer.
    
    Args:
        df (pd.DataFrame): Input transaction dataframe
        customer_id_col (str): Name of customer ID column
        date_col (str): Name of date column
        invoice_col (str): Name of invoice column
        price_col (str): Name of price column
        quantity_col (str): Name of quantity column
        product_col (str): Name of product column
        country_col (str): Name of country column
        
    Returns:
        pd.DataFrame: RFM features for each customer
    """
    # Reference date (day after last purchase)
    reference_date = df[date_col].max() + dt.timedelta(days=1)
    
    # RFM Features
    customer_data = df.groupby(customer_id_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        invoice_col: 'nunique',  # Frequency
        price_col: 'sum',  # Monetary
        quantity_col: 'sum',  # Total items
        product_col: 'nunique',  # Product variety
        country_col: 'first'  # Primary country
    }).rename(columns={
        date_col: 'Recency',
        invoice_col: 'Frequency',
        price_col: 'Monetary',
        quantity_col: 'TotalItems',
        product_col: 'UniqueProducts',
        country_col: 'Country'
    })
    
    customer_data.reset_index(inplace=True)
    return customer_data


def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional behavioral features.
    
    Args:
        df (pd.DataFrame): RFM dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    df = df.copy()
    
    # Average Order Value
    df['AvgOrderValue'] = df['Monetary'] / df['Frequency']
    
    # Items per Order
    df['ItemsPerOrder'] = df['TotalItems'] / df['Frequency']
    
    return df


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for customer segmentation.
    
    Args:
        df (pd.DataFrame): Preprocessed transaction dataframe
        
    Returns:
        pd.DataFrame: Customer features dataframe
    """
    # Prepare date column
    df = prepare_date_column(df)
    
    # Calculate total price
    df = calculate_total_price(df)
    
    # Calculate RFM features
    customer_data = calculate_rfm_features(df)
    
    # Calculate additional features
    customer_data = calculate_additional_features(customer_data)
    
    return customer_data


def scale_features(df: pd.DataFrame, 
                  feature_columns: List[str],
                  scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features for clustering.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (List[str]): List of columns to scale
        scaler (StandardScaler, optional): Pre-fitted scaler
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled features and fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
    else:
        scaled_features = scaler.transform(df[feature_columns])
    
    # Create dataframe with scaled features
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=[f"{col}_scaled" for col in feature_columns],
        index=df.index
    )
    
    return scaled_df, scaler


def get_default_feature_columns() -> List[str]:
    """
    Get default feature columns for clustering.
    
    Returns:
        List[str]: List of default feature columns
    """
    return ['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 
            'AvgOrderValue', 'ItemsPerOrder']


def get_rfm_feature_columns() -> List[str]:
    """
    Get RFM feature columns for clustering.
    
    Returns:
        List[str]: List of RFM feature columns
    """
    return ['Recency', 'Frequency', 'Monetary']


def save_customer_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save customer features to CSV file.
    
    Args:
        df (pd.DataFrame): Customer features dataframe
        output_path (str): Path to save the features
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    input_path = "../data/processed/Online_Retail_Cleaned.csv"
    output_path = "../data/processed/Customer_RFM_Features.csv"
    
    # Load processed data
    df = pd.read_csv(input_path)
    
    # Create customer features
    customer_features = create_customer_features(df)
    
    # Save features
    save_customer_features(customer_features, output_path)
    
    print(f"Feature engineering completed. Shape: {customer_features.shape}")
    print(f"Customer features saved to: {output_path}")
    
    # Display feature summary
    print("\nFeature Summary:")
    print(customer_features.describe())
