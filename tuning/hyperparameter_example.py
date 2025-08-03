import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.abspath('..'))

# Import our hyperparameter tuner
from tuning.hyperparameter_tuning import OptunaHyperparameterTuner

def main():
    """Example of using the OptunaHyperparameterTuner with a real dataset"""
    
    # Load data (adjust path as needed)
    print("Loading data...")
    try:
        df = pd.read_csv("../c1_train.csv")
    except FileNotFoundError:
        print("Error: Dataset not found. Please make sure the file exists.")
        return
    
    # Convert date column to datetime
    df['effectivedate'] = pd.to_datetime(df['effectivedate'])
    
    # Simple data preparation
    print("Preparing data...")
    # Filter to just one plant for simplicity
    plant_id = 1  # You can change this to any available plant
    df_plant = df[df['plant_name'] == plant_id].copy()
    
    # Select features and target
    target_column = 'production'
    date_column = 'effectivedate'
    
    # Select a subset of numerical features for simplicity
    features = [
        'provider1_fc0', 'provider1_fc1200', 'provider2_fc0', 'provider2_fc1200',
        'hour', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos', 'capacity'
    ]
    
    # Create train/test split
    # Use last 30 days as test set
    split_date = df_plant['effectivedate'].max() - pd.Timedelta(days=30)
    
    train_df = df_plant[df_plant['effectivedate'] < split_date]
    test_df = df_plant[df_plant['effectivedate'] >= split_date]
    
    X_train = train_df[features]
    y_train = train_df[target_column].values
    X_test = test_df[features]
    y_test = test_df[target_column].values
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Example 1: Tree-based model tuning
    print("\n--- Example 1: LightGBM Model Tuning ---")
    tune_lgbm(X_train, y_train, X_test, y_test)
    
    # Example 2: Time-series model tuning
    print("\n--- Example 2: Prophet Model Tuning ---")
    tune_prophet(train_df, test_df, date_column, target_column)
    
    print("\nHyperparameter tuning examples completed.")

def tune_lgbm(X_train, y_train, X_test, y_test, n_trials=20, case_name="lgbm_test"):
    """Tune LightGBM model hyperparameters"""
    
    # Initialize tuner with reduced number of trials for demonstration
    tuner = OptunaHyperparameterTuner(
        model_type='lgbm',
        metric='rmse',
        n_trials=n_trials,  # Reduced for demonstration
        cv=3,  # Reduced for faster execution
        random_state=42
    )
    
    # Tune hyperparameters and save with case name
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    best_params = tuner.tune(X_train, y_train, case_name=case_name)
    print(f"Best parameters: {best_params}")
    print(f"Parameters saved to best_params/{tuner.model_type}_{case_name}.json")
    
    # Get best model
    best_model = tuner.get_best_model()
    
    # Train and evaluate
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE with tuned hyperparameters: {rmse:.4f}")
    
    # Compare with default parameters
    default_model = tuner.get_model_with_default_params()
    default_model.fit(X_train, y_train)
    y_pred_default = default_model.predict(X_test)
    rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
    print(f"RMSE with default hyperparameters: {rmse_default:.4f}")
    print(f"Improvement: {(rmse_default - rmse) / rmse_default * 100:.2f}%")
    
    # Demonstrate loading parameters from file
    print("\nDemonstrating loading parameters from file:")
    new_tuner = OptunaHyperparameterTuner(model_type='lgbm')
    loaded_params = new_tuner.load_best_params(case_name)
    if loaded_params:
        print(f"Successfully loaded parameters: {loaded_params}")
        loaded_model = new_tuner.get_best_model()
        loaded_model.fit(X_train, y_train)
        y_pred_loaded = loaded_model.predict(X_test)
        rmse_loaded = np.sqrt(mean_squared_error(y_test, y_pred_loaded))
        print(f"RMSE with loaded parameters: {rmse_loaded:.4f}")
    
    # Plot predictions vs actual for the first 100 test points
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual')
    plt.plot(y_pred[:100], label='Tuned Model Prediction')
    plt.plot(y_pred_default[:100], label='Default Model Prediction')
    plt.title('Actual vs Predicted Values (LGBM)')
    plt.legend()
    plt.savefig('lgbm_predictions.png')
    print("Plot saved as 'lgbm_predictions.png'")
    
    return best_params, rmse

def tune_prophet(train_df, test_df, date_column, target_column, n_trials=20, case_name="prophet_test"):
    """Tune Prophet model hyperparameters"""
    
    # Add a helper to get the default model
    def get_default_prophet():
        from models.regression_models import ProphetModel
        return ProphetModel()
    
    # Initialize tuner
    tuner = OptunaHyperparameterTuner(
        model_type='prophet',
        metric='rmse',
        n_trials=n_trials,  # Reduced for demonstration
        cv=3,  # Reduced for faster execution
        random_state=42
    )
    
    # Add get_model_with_default_params method to the tuner
    tuner.get_model_with_default_params = get_default_prophet
    
    # Prepare minimal dataset for Prophet
    prophet_train = train_df[[date_column, target_column]].copy()
    prophet_test = test_df[[date_column, target_column]].copy()
    
    # Tune hyperparameters with case name
    print(f"Starting Prophet hyperparameter tuning with {n_trials} trials...")
    best_params = tuner.tune(
        prophet_train, 
        date_column=date_column, 
        target_column=target_column,
        case_name=case_name
    )
    print(f"Best parameters: {best_params}")
    print(f"Parameters saved to best_params/{tuner.model_type}_{case_name}.json")
    
    # Get best model
    best_model = tuner.get_best_model()
    
    # Train and evaluate
    best_model.fit(prophet_train, date_column, target_column)
    y_pred = best_model.predict(prophet_test, date_column)
    y_test = prophet_test[target_column].values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE with tuned hyperparameters: {rmse:.4f}")
    
    # Compare with default parameters
    default_model = get_default_prophet()
    default_model.fit(prophet_train, date_column, target_column)
    y_pred_default = default_model.predict(prophet_test, date_column)
    rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
    print(f"RMSE with default hyperparameters: {rmse_default:.4f}")
    print(f"Improvement: {(rmse_default - rmse) / rmse_default * 100:.2f}%")
    
    # Demonstrate loading parameters from file
    print("\nDemonstrating loading parameters from file:")
    new_tuner = OptunaHyperparameterTuner(model_type='prophet')
    loaded_params = new_tuner.load_best_params(case_name)
    if loaded_params:
        print(f"Successfully loaded parameters: {loaded_params}")
        loaded_model = new_tuner.get_best_model()
        loaded_model.fit(prophet_train, date_column, target_column)
        y_pred_loaded = loaded_model.predict(prophet_test, date_column)
        rmse_loaded = np.sqrt(mean_squared_error(y_test, y_pred_loaded))
        print(f"RMSE with loaded parameters: {rmse_loaded:.4f}")
    
    # Plot predictions vs actual for a sample period
    plt.figure(figsize=(12, 6))
    
    # Get a subset of data for the plot
    sample_size = min(100, len(y_test))
    dates = prophet_test[date_column].iloc[:sample_size]
    actuals = y_test[:sample_size]
    preds = y_pred[:sample_size]
    default_preds = y_pred_default[:sample_size]
    
    plt.plot(dates, actuals, label='Actual')
    plt.plot(dates, preds, label='Tuned Model Prediction')
    plt.plot(dates, default_preds, label='Default Model Prediction')
    plt.title('Actual vs Predicted Values (Prophet)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prophet_predictions.png')
    print("Plot saved as 'prophet_predictions.png'")
    
    return best_params, rmse

if __name__ == "__main__":
    main() 