import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Path to the trained model
MODEL_PATH = Path(__file__).parent.parent / "model" / "xgb_numeric_only.pkl"


def bool_to_int_helper(x):
    """Convert boolean values to integers (0/1)"""
    return x.astype(int)


def load_model():
    """
    Load the trained XGBoost model pipeline from disk.
    
    Returns:
        Loaded sklearn Pipeline object containing preprocessor and classifier
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    print(f"Loading model from {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def predict_data(input_df, model=None):
    """
    Make predictions on input data using the trained model.
    
    The model expects a DataFrame with the following numeric features:
    - age
    - alcohol_consumption_per_week
    - physical_activity_minutes_per_week
    - diet_score
    - sleep_hours_per_day
    - screen_time_hours_per_day
    - bmi
    - waist_to_hip_ratio
    - systolic_bp
    - diastolic_bp
    - heart_rate
    - cholesterol_total
    - hdl_cholesterol
    - ldl_cholesterol
    - triglycerides
    - family_history_diabetes
    - hypertension_history
    - cardiovascular_history
    
    Categorical features (gender, ethnicity, education_level, income_level, 
    smoking_status, employment_status) will be automatically dropped by the pipeline.
    
    Args:
        input_df: pandas DataFrame containing input features (with or without 'id' column)
        model: Optional pre-loaded model. If None, will load from MODEL_PATH
        
    Returns:
        pandas DataFrame with columns ['id', 'probability'] containing predictions
        
    Raises:
        ValueError: If prediction fails due to data issues
        FileNotFoundError: If model file doesn't exist
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Check if 'id' column exists
    has_id = 'id' in input_df.columns
    
    if has_id:
        # Extract ID column
        id_col = 'id'
        ids = input_df[id_col].copy()
        
        # Drop ID column for prediction
        X_new = input_df.drop(columns=[id_col])
    else:
        # No ID column, create sequential IDs
        ids = pd.Series(range(len(input_df)), name='id')
        X_new = input_df.copy()
    
    try:
        # Make predictions (get probability of class 1)
        probs = model.predict_proba(X_new)[:, 1]
    except ValueError as e:
        raise ValueError(f"Error during prediction: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during prediction: {e}")
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'id': ids,
        'probability': probs
    })
    
    return result_df


def predict_single(data_dict, model=None):
    """
    Make a prediction for a single data point.
    
    Args:
        data_dict: Dictionary containing feature values
        model: Optional pre-loaded model. If None, will load from MODEL_PATH
        
    Returns:
        Dictionary with 'id' (if provided) and 'probability'
        
    Example:
        data = {
            'age': 45,
            'bmi': 28.5,
            'systolic_bp': 130,
            # ... other features
        }
        result = predict_single(data)
        # Returns: {'probability': 0.65}
    """
    # Convert single data point to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Make prediction
    result_df = predict_data(df, model=model)
    
    # Convert to dictionary
    result = result_df.iloc[0].to_dict()
    
    return result


def predict_batch(data_list, model=None):
    """
    Make predictions for a batch of data points.
    
    Args:
        data_list: List of dictionaries, each containing feature values
        model: Optional pre-loaded model. If None, will load from MODEL_PATH
        
    Returns:
        List of dictionaries with 'id' (if provided) and 'probability'
        
    Example:
        data = [
            {'age': 45, 'bmi': 28.5, ...},
            {'age': 52, 'bmi': 31.2, ...},
        ]
        results = predict_batch(data)
        # Returns: [{'id': 0, 'probability': 0.65}, {'id': 1, 'probability': 0.72}]
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    
    # Make predictions
    result_df = predict_data(df, model=model)
    
    # Convert to list of dictionaries
    results = result_df.to_dict('records')
    
    return results
