"""
========================================
PRODUCTIVITY PREDICTION SYSTEM - ML CODE
========================================
Complete Machine Learning Implementation
Uses Linear Regression + Random Forest
Dataset: Student Productivity Data

FIXED VERSION - Path issue resolved
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

# ========================================
# 1. LOAD AND EXPLORE DATA
# ========================================

def load_data(filepath):
    """
    Load the Kaggle dataset
    
    Parameters:
    filepath (str): Path to CSV file
    
    Returns:
    DataFrame: Loaded data
    """
    print("="*60)
    print("LOADING DATA...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Data loaded successfully!")
    print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData Info:")
    print(df.info())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    return df

# ========================================
# 2. DATA PREPROCESSING
# ========================================

def preprocess_data(df):
    """
    Preprocess the dataset
    - Handle missing values
    - Select relevant features
    - Create productivity score if needed
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING...")
    print("="*60)
    
    # Make a copy
    df_processed = df.copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype in ['float64', 'int64']:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                print(f"✓ Filled {col} with mean value")
            else:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                print(f"✓ Filled {col} with mode value")
    
    # Select numeric features
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    
    # If 'Sleep_Duration' and 'Study_Hours' exist, use them directly
    features_to_use = []
    for feature in ['Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Physical_Activity', 'Caffeine_Intake']:
        if feature in numeric_cols:
            features_to_use.append(feature)
    
    if len(features_to_use) < 3:
        print("\nWarning: Expected features not found. Using all numeric columns.")
        features_to_use = numeric_cols
    
    print(f"✓ Features to use: {features_to_use}")
    
    # Create target variable if not exists
    if 'Sleep_Quality' in numeric_cols:
        target = 'Sleep_Quality'
    elif 'Academic_Performance' in numeric_cols:
        target = 'Academic_Performance'
    elif 'Grades' in numeric_cols:
        target = 'Grades'
    else:
        # Create synthetic productivity score
        print("\nCreating synthetic productivity score...")
        target = 'Productivity_Score'
        df_processed[target] = (
            df_processed[features_to_use[0]] * 0.3 +
            df_processed[features_to_use[1]] * 0.3 -
            df_processed[features_to_use[2]] * 0.2 +
            (df_processed[features_to_use[3]] if len(features_to_use) > 3 else 0) * 0.2
        )
    
    print(f"✓ Target variable: {target}")
    
    return df_processed, features_to_use, target

# ========================================
# 3. FEATURE ENGINEERING
# ========================================

def feature_engineering(df, features):
    """
    Create new features from existing ones
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING...")
    print("="*60)
    
    df_eng = df.copy()
    
    # Normalize features to 0-1 range
    scaler = MinMaxScaler()
    df_eng[features] = scaler.fit_transform(df_eng[features])
    
    print(f"✓ Features normalized to 0-1 range")
    
    # Create interaction features if enough columns
    if len(features) >= 2:
        df_eng['sleep_study_ratio'] = (df_eng[features[0]] + 1) / (df_eng[features[1]] + 1)
        print(f"✓ Created: sleep_study_ratio")
    
    if len(features) >= 3:
        df_eng['activity_screen_ratio'] = (df_eng[features[3]] if len(features) > 3 else 1) / (df_eng[features[2]] + 1)
        print(f"✓ Created: activity_screen_ratio")
    
    print(f"✓ Feature engineering complete!")
    
    return df_eng

# ========================================
# 4. EXPLORATORY DATA ANALYSIS
# ========================================

def exploratory_analysis(df, features, target):
    """
    Perform EDA and create visualizations
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS...")
    print("="*60)
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df[features + [target]].select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_heatmap.png")
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:6]):
        if feature in df.columns:
            axes[idx].hist(df[feature].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_distributions.png")
    plt.close()
    
    print(f"\nCorrelation with target ({target}):")
    if target in correlation.columns:
        print(correlation[target].sort_values(ascending=False))
    
    print(f"✓ EDA complete!")

# ========================================
# 5. MODEL TRAINING
# ========================================

def train_models(X, y):
    """
    Train multiple ML models
    
    Returns:
    dict: Trained models and metrics
    """
    print("\n" + "="*60)
    print("TRAINING MODELS...")
    print("="*60)
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features standardized")
    
    results = {}
    
    # ===== MODEL 1: LINEAR REGRESSION =====
    print("\n--- Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    print(f"R² Score: {lr_r2:.4f}")
    print(f"MAE: {lr_mae:.4f}")
    print(f"RMSE: {lr_rmse:.4f}")
    
    results['linear_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'r2': lr_r2,
        'mae': lr_mae,
        'rmse': lr_rmse,
        'predictions': y_pred_lr
    }
    
    # ===== MODEL 2: RANDOM FOREST =====
    print("\n--- Random Forest Regression ---")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    print(f"R² Score: {rf_r2:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print(f"RMSE: {rf_rmse:.4f}")
    
    results['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'r2': rf_r2,
        'mae': rf_mae,
        'rmse': rf_rmse,
        'predictions': y_pred_rf,
        'feature_importance': rf_model.feature_importances_
    }
    
    # Choose best model
    best_model = 'random_forest' if rf_r2 > lr_r2 else 'linear_regression'
    print(f"\n✓ Best Model: {best_model.upper()} (R²: {max(rf_r2, lr_r2):.4f})")
    
    return results, X_train, X_test, y_train, y_test, scaler

# ========================================
# 6. MODEL EVALUATION & VISUALIZATION
# ========================================

def evaluate_models(results, y_test):
    """
    Evaluate and compare models
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Actual vs Predicted (Linear Regression)
    axes[0].scatter(y_test, results['linear_regression']['predictions'], 
                   alpha=0.6, color='blue', label='Linear Regression')
    axes[0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Linear Regression: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Random Forest)
    axes[1].scatter(y_test, results['random_forest']['predictions'], 
                   alpha=0.6, color='green', label='Random Forest')
    axes[1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('Random Forest: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()
    
    # Model comparison table
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}")
    print("-" * 60)
    print(f"{'Linear Regression':<20} {results['linear_regression']['r2']:<15.4f} {results['linear_regression']['mae']:<15.4f} {results['linear_regression']['rmse']:<15.4f}")
    print(f"{'Random Forest':<20} {results['random_forest']['r2']:<15.4f} {results['random_forest']['mae']:<15.4f} {results['random_forest']['rmse']:<15.4f}")
    print("-" * 60)

# ========================================
# 7. FEATURE IMPORTANCE
# ========================================

def plot_feature_importance(results, features):
    """
    Plot feature importance from Random Forest
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE...")
    print("="*60)
    
    importances = results['random_forest']['feature_importance']
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()
    
    print("\nFeature Importance Ranking:")
    for idx, feature_idx in enumerate(indices):
        print(f"{idx+1}. {features[feature_idx]}: {importances[feature_idx]:.4f}")

# ========================================
# 8. SAVE MODELS
# ========================================

def save_models(results, features):
    """
    Save trained models to files
    """
    print("\n" + "="*60)
    print("SAVING MODELS...")
    print("="*60)
    
    # Save Linear Regression
    joblib.dump(results['linear_regression']['model'], 'linear_regression_model.pkl')
    joblib.dump(results['linear_regression']['scaler'], 'linear_regression_scaler.pkl')
    print("✓ Saved: linear_regression_model.pkl")
    print("✓ Saved: linear_regression_scaler.pkl")
    
    # Save Random Forest
    joblib.dump(results['random_forest']['model'], 'random_forest_model.pkl')
    print("✓ Saved: random_forest_model.pkl")
    
    # Save features list
    joblib.dump(features, 'features_list.pkl')
    print("✓ Saved: features_list.pkl")
    
    # Save metadata
    metadata = {
        'lr_r2': results['linear_regression']['r2'],
        'rf_r2': results['random_forest']['r2'],
        'features': features,
        'feature_importance': results['random_forest']['feature_importance'].tolist()
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✓ Saved: model_metadata.json")

# ========================================
# 9. MAKE PREDICTIONS
# ========================================

def make_prediction(sleep, study, screen, activity, caffeine):
    """
    Make prediction on new data
    
    Parameters:
    sleep, study, screen, activity, caffeine (float): Input values
    
    Returns:
    dict: Prediction result
    """
    # Load model
    rf_model = joblib.load('random_forest_model.pkl')
    
    # Create input
    input_data = np.array([[sleep, study, screen, activity, caffeine]])
    
    # Predict
    prediction = rf_model.predict(input_data)[0]
    
    # Normalize to 0-100
    prediction = max(0, min(100, prediction))
    
    return {
        'score': round(prediction, 2),
        'level': 'HIGH' if prediction >= 67 else ('MEDIUM' if prediction >= 34 else 'LOW')
    }

# ========================================
# 10. MAIN EXECUTION
# ========================================

def main():
    """
    Main execution function
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "PRODUCTIVITY PREDICTION SYSTEM" + " "*13 + "║")
    print("║" + " "*18 + "Machine Learning Pipeline" + " "*15 + "║")
    print("╚" + "="*58 + "╝")
    
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Build the full path to the CSV file
    CSV_FILE = os.path.join(BASE_DIR, 'student_sleep_patterns.csv')
    
    print(f"\nLooking for CSV file at: {CSV_FILE}")
    
    try:
        # 1. Load data
        df = load_data(CSV_FILE)
        
        # 2. Preprocess
        df_processed, features, target = preprocess_data(df)
        
        # 3. Feature engineering
        df_eng = feature_engineering(df_processed, features)
        
        # 4. EDA
        exploratory_analysis(df_eng, features, target)
        
        # 5. Prepare data for modeling
        X = df_eng[features]
        y = df_eng[target]
        
        # 6. Train models
        results, X_train, X_test, y_train, y_test, scaler = train_models(X, y)
        
        # 7. Evaluate
        evaluate_models(results, y_test)
        
        # 8. Feature importance
        plot_feature_importance(results, features)
        
        # 9. Save models
        save_models(results, features)
        
        # 10. Test prediction
        print("\n" + "="*60)
        print("TEST PREDICTION...")
        print("="*60)
        prediction = make_prediction(8, 6, 2.5, 45, 1)
        print(f"Input: Sleep=8h, Study=6h, Screen=2.5h, Activity=45min, Caffeine=1cup")
        print(f"Prediction: {prediction['score']}/100 - {prediction['level']} PRODUCTIVITY")
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE!")
        print("="*60)
        print("\nGenerated Files:")
        print("  - correlation_heatmap.png")
        print("  - feature_distributions.png")
        print("  - model_comparison.png")
        print("  - feature_importance.png")
        print("  - linear_regression_model.pkl")
        print("  - linear_regression_scaler.pkl")
        print("  - random_forest_model.pkl")
        print("  - features_list.pkl")
        print("  - model_metadata.json")
        print("\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found!")
        print(f"Expected location: {CSV_FILE}")
        print(f"\nPlease ensure:")
        print("1. The file 'student_sleep_patterns.csv' exists")
        print("2. It is in the same folder as main.py")
        print("3. The file name is spelled correctly (case-sensitive)")
        print(f"\nError details: {str(e)}")

if __name__ == "__main__":
    main()