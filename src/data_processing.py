import pandas as pd
import numpy as np

def preprocess_data(df, is_training=True):
    """
    Preprocess the heart disease data
    
    Args:
        df: Input dataframe
        is_training: Whether this is training data (with cardio column) or scoring data
    """
    df = df.copy()
    
    # Convert categorical columns
    categorical_col = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
    if is_training:
        categorical_col.append('cardio')
    df[categorical_col] = df[categorical_col].astype('category')
    
    # Convert age from days to years
    if 'age' in df.columns:  # Check if age exists
        average_days_per_year = 365.25
        df['age'] = (df['age'] / average_days_per_year).round().astype(int)
    
    # Calculate BMI if height and weight exist
    if 'height' in df.columns and 'weight' in df.columns:
        df["bmi"] = (df["weight"] / ((df["height"]/100) ** 2)).round(2)
        # Remove height and weight after BMI calculation
        df.pop('height')
        df.pop('weight')
    
    # Remove ID if exists
    if 'id' in df.columns:
        df.pop('id')
    
    # Reorder columns for consistency
    expected_columns = ["age", "gender", "bmi", "ap_hi", "ap_lo", 
                       "cholesterol", "gluc", "smoke", "alco", "active"]
    
    if is_training:
        expected_columns.append("cardio")
    
    # Only select columns that exist in the dataframe
    existing_columns = [col for col in expected_columns if col in df.columns]
    df = df[existing_columns]
    
    return df 