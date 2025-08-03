import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(predictions_df, true_column, prediction_columns, date_column=None):
    """
    Calculate evaluation metrics for multiple models.
    
    Parameters:
        predictions_df (DataFrame): DataFrame containing true values and predictions
        true_column (str): Name of the column with true values
        prediction_columns (list): List of column names with model predictions
        date_column (str, optional): Name of the date column (not used in calculations but included for consistency)
        
    Returns:
        DataFrame: DataFrame with metrics for each model
    """
    metrics_dict = {}
    
    for col in prediction_columns:
        y_true = predictions_df[true_column].values
        y_pred = predictions_df[col].values
        
        # Calculate basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE and MAAPE: Avoid division by zero by only computing on non-zero true values
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
            maape = np.mean(np.arctan(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])))
        else:
            mape = np.nan
            maape = np.nan
        
        # sMAPE: Avoid division by zero in the denominator
        denominator = np.abs(y_true) + np.abs(y_pred)
        nonzero_denom_mask = denominator != 0
        if np.any(nonzero_denom_mask):
            smape = np.mean(2 * np.abs(y_pred[nonzero_denom_mask] - y_true[nonzero_denom_mask]) / denominator[nonzero_denom_mask]) * 100
        else:
            smape = np.nan
        
        # MASE: Use the naive forecast (lag 1) errors; if their mean is zero, set MASE to np.nan
        naive_errors = np.abs(np.diff(y_true))
        mean_naive_error = np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) != 0 else np.nan
        mase = mae / mean_naive_error if not np.isnan(mean_naive_error) and mean_naive_error != 0 else np.nan
        
        metrics_dict[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MAAPE': maape,
            'sMAPE': smape,
            'MASE': mase
        }
    
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Model'})
    return metrics_df

def calculate_prediction_errors(predictions_df, true_column, prediction_columns):
    """
    Calculate prediction errors for multiple models.
    
    Parameters:
        predictions_df (DataFrame): DataFrame containing true values and predictions
        true_column (str): Name of the column with true values
        prediction_columns (list): List of column names with model predictions
        
    Returns:
        DataFrame: DataFrame with error values for each model
    """
    errors_df = predictions_df.copy()
    y_true = errors_df[true_column]
    
    for col in prediction_columns:
        # Calculate various error metrics
        errors_df[f'{col}_error'] = errors_df[col] - y_true
        errors_df[f'{col}_abs_error'] = abs(errors_df[col] - y_true)
        
        # Calculate percentage errors (avoiding division by zero)
        nonzero_mask = y_true != 0
        errors_df[f'{col}_pct_error'] = np.nan
        errors_df.loc[nonzero_mask, f'{col}_pct_error'] = (
            (errors_df.loc[nonzero_mask, col] - y_true[nonzero_mask]) / y_true[nonzero_mask] * 100
        )
        
        # Calculate squared errors
        errors_df[f'{col}_squared_error'] = errors_df[f'{col}_error'] ** 2
    
    return errors_df

def summarize_errors_by_period(errors_df, date_column, prediction_columns, period='D'):
    """
    Summarize errors by time period (e.g., day, month).
    
    Parameters:
        errors_df (DataFrame): DataFrame with error values from calculate_prediction_errors
        date_column (str): Name of the date column
        prediction_columns (list): List of column names with model predictions
        period (str): Period to group by ('D' for day, 'M' for month, etc.)
        
    Returns:
        DataFrame: DataFrame with summarized errors by period
    """
    # Ensure the date column is datetime
    errors_df[date_column] = pd.to_datetime(errors_df[date_column])
    
    # Create a period column
    if period == 'D':
        errors_df['period'] = errors_df[date_column].dt.date
    elif period == 'M':
        errors_df['period'] = errors_df[date_column].dt.to_period('M')
    elif period == 'W':
        errors_df['period'] = errors_df[date_column].dt.to_period('W')
    elif period == 'Y':
        errors_df['period'] = errors_df[date_column].dt.year
    elif period == 'H':
        errors_df['period'] = errors_df[date_column].dt.floor('H')
    else:
        errors_df['period'] = errors_df[date_column].dt.to_period(period)
    
    # Group by period and calculate mean metrics
    summary = []
    
    for model in prediction_columns:
        # Calculate mean errors for each period
        period_summary = errors_df.groupby('period').agg({
            f'{model}_error': ['mean', 'std'],
            f'{model}_abs_error': 'mean',
            f'{model}_pct_error': ['mean', 'median'],
            f'{model}_squared_error': 'mean'
        })
        
        # Flatten multi-level columns
        period_summary.columns = [f'{model}_{col[0]}_{col[1]}' if col[1] else f'{model}_{col[0]}' 
                              for col in period_summary.columns]
        
        # Add RMSE
        period_summary[f'{model}_rmse'] = np.sqrt(period_summary[f'{model}_squared_error_mean'])
        
        summary.append(period_summary)
    
    # Combine all model summaries
    combined_summary = pd.concat(summary, axis=1)
    return combined_summary.reset_index()

def compare_models_across_plants(results_dict, metric='MAE'):
    """
    Compare models across different plants for a specific metric.
    
    Parameters:
        results_dict (dict): Dictionary with plant names as keys and metric DataFrames as values
        metric (str): Metric to compare ('MAE', 'RMSE', etc.)
        
    Returns:
        DataFrame: DataFrame with the specified metric for each model across all plants
    """
    comparison = {}
    
    for plant, metrics_df in results_dict.items():
        # Create a dictionary with model names as keys and the specified metric as values
        plant_metrics = dict(zip(metrics_df['Model'], metrics_df[metric]))
        comparison[f'Plant_{plant}'] = plant_metrics
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison).T
    
    # Add average across plants
    comparison_df['Average'] = comparison_df.mean(axis=1)
    
    return comparison_df

if __name__ == "__main__":
    # Example usage
    import numpy as np
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    true_values = np.sin(np.linspace(0, 10, 100)) * 10 + 20
    
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual': true_values,
        'model1': true_values + np.random.normal(0, 1, 100),
        'model2': true_values + np.random.normal(0, 2, 100)
    })
    
    # Evaluate models
    metrics = evaluate_models(predictions_df, 'actual', ['model1', 'model2'], 'date')
    print("Metrics:")
    print(metrics)
    
    # Calculate errors
    errors = calculate_prediction_errors(predictions_df, 'actual', ['model1', 'model2'])
    print("\nErrors (first 5 rows):")
    print(errors.head())
