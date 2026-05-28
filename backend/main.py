"""
========================================
PRODUCTIVITY PREDICTION SYSTEM - ML CODE
========================================
Complete Machine Learning Implementation
Uses Random Forest Regressor for robust prediction & feature importance.
Dataset: Student Productivity Data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

# ========================================
# 1. LOAD AND EXPLORE DATA
# ========================================

def load_data(filepath):
    """ Load the Kaggle dataset """
    print("="*60)
    print("LOADING DATA...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    # Check for the expected target column
    if 'Sleep_Quality' not in df.columns:
        raise ValueError("The 'Sleep_Quality' column is missing from the CSV file.")

    print(f"✓ Data loaded successfully!")
    return df

# ========================================
# 2. DATA PREPROCESSING & TARGET SCALING
# ========================================

def preprocess_data(df):
    """
    Preprocess the dataset
    - Select relevant features
    - Scale the target 'Sleep_Quality' (1-10) to 'Productivity_Score' (10-100)
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING...")
    print("="*60)
    
    df_processed = df.copy()
    
    # Define features and target (using a simplified set for the front-end inputs)
    features_to_use = ['Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Physical_Activity', 'Caffeine_Intake']
    target = 'Productivity_Score'
    
    # Check if all required features exist
    if not all(feature in df_processed.columns for feature in features_to_use):
        raise ValueError(f"Missing one or more required features: {features_to_use}")

    # Scale Sleep_Quality (1-10) to Productivity_Score (10-100)
    # The max possible Sleep_Quality is 10, so max score is 100
    df_processed[target] = df_processed['Sleep_Quality'] * 10 
    
    # Convert Physical_Activity from minutes to hours for consistency with other features
    df_processed['Physical_Activity'] = df_processed['Physical_Activity'] / 60 
    
    print(f"✓ Target variable '{target}' created by scaling 'Sleep_Quality' (10-100)")
    print(f"✓ 'Physical_Activity' converted to hours.")
    print(f"✓ Features to use: {features_to_use}")
    
    # Keep only the columns we need for the ML model (using the scaled activity)
    df_model = df_processed[features_to_use + [target]].copy()
    
    # Handle any potential NaN values by dropping them (since the provided dataset is clean)
    df_model.dropna(inplace=True)
    
    return df_model, features_to_use, target

# ========================================
# 3. MODEL TRAINING
# ========================================

def train_models(X, y):
    """ Train Random Forest Regressor and Linear Regression """
    print("\n" + "="*60)
    print("TRAINING MODELS...")
    print("="*60)
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing")
    
    results = {}
    
    # --- MODEL 1: LINEAR REGRESSION --- (Needs scaling)
    print("\n--- Linear Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, y_pred_lr)
    print(f"R² Score: {lr_r2:.4f}")
    
    results['linear_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'r2': lr_r2,
        'predictions': y_pred_lr
    }
    
    # --- MODEL 2: RANDOM FOREST --- (Does not strictly require scaling, generally better)
    print("\n--- Random Forest Regression ---")
    rf_model = RandomForestRegressor(
        n_estimators=200, # Increased for better performance
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    print(f"R² Score: {rf_r2:.4f}")
    
    results['random_forest'] = {
        'model': rf_model,
        'r2': rf_r2,
        'predictions': y_pred_rf,
        'feature_importance': rf_model.feature_importances_
    }
    
    return results, X_train.columns.tolist()

# ========================================
# 4. SAVE MODELS
# ========================================

def save_models(results, features):
    """ Save trained models and metadata to files """
    print("\n" + "="*60)
    print("SAVING MODELS...")
    print("="*60)
    
    # Save Random Forest (The primary model for the Flask API)
    joblib.dump(results['random_forest']['model'], 'random_forest_model.pkl')
    print("✓ Saved: random_forest_model.pkl")

    # Save Linear Regression and its scaler (As backup/reference)
    joblib.dump(results['linear_regression']['model'], 'linear_regression_model.pkl')
    joblib.dump(results['linear_regression']['scaler'], 'linear_regression_scaler.pkl')
    print("✓ Saved: linear_regression_model.pkl and linear_regression_scaler.pkl")
    
    # Save features list
    joblib.dump(features, 'features_list.pkl')
    print("✓ Saved: features_list.pkl")
    
    # Prepare feature importance for API
    feature_importance_dict = {f: float(results['random_forest']['feature_importance'][i]) 
                               for i, f in enumerate(features)}
    
    # Save metadata
    metadata = {
        'lr_r2': float(results['linear_regression']['r2']),
        'rf_r2': float(results['random_forest']['r2']),
        'features': features,
        'feature_importance': feature_importance_dict
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✓ Saved: model_metadata.json")

# ========================================
# 5. MAIN EXECUTION
# ========================================

def main():
    """ Main execution function """
    
    # ... (omitted boilerplate text for brevity)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(BASE_DIR, 'student_sleep_patterns.csv')
    
    try:
        # 1. Load and Preprocess Data
        df = load_data(CSV_FILE)
        df_model, features, target = preprocess_data(df)
        
        X = df_model[features]
        y = df_model[target]
        
        # 2. Train Models
        results, feature_names = train_models(X, y)
        
        # 3. Save Models
        save_models(results, feature_names)
        
        # 4. Test prediction
        print("\n" + "="*60)
        print("TEST PREDICTION...")
        print("="*60)
        # Test input: Sleep=8h, Study=6h, Screen=2.5h, Activity=45min (0.75h), Caffeine=1cup
        rf_model = joblib.load('random_forest_model.pkl')
        test_input = np.array([[8.0, 6.0, 2.5, 0.75, 1.0]])
        prediction = rf_model.predict(test_input)[0]
        score = max(0.0, min(100.0, prediction))
        
        print(f"Input: Sleep=8h, Study=6h, Screen=2.5h, Activity=0.75h, Caffeine=1cup")
        print(f"Prediction: {score:.1f}/100 - {'HIGH' if score >= 67 else ('MEDIUM' if score >= 34 else 'LOW')} PRODUCTIVITY")
        
        print("\n" + "="*60)
        print("✓ ML PIPELINE COMPLETE!")
        print("✓ Models trained and saved. Start 'app.py' now.")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at {CSV_FILE}")
        print("Please ensure 'student_sleep_patterns.csv' is in the correct directory.")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()