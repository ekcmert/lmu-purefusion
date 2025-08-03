import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_predictions(predictions_df, date_column, true_column, prediction_columns, 
                          title="True vs. Predicted Values"):
    """
    Create a line plot comparing true values and predictions over time.
    
    Parameters:
        predictions_df (DataFrame): DataFrame containing true values and predictions
        date_column (str): Name of the date column
        true_column (str): Name of the column with true values
        prediction_columns (list): List of column names with model predictions
        title (str): Title for the plot
        
    Returns:
        Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add true values
    fig.add_trace(go.Scatter(
        x=predictions_df[date_column], 
        y=predictions_df[true_column],
        mode='lines', 
        name='True', 
        line=dict(width=3, color='black')
    ))
    
    # Add predictions for each model
    for col in prediction_columns:
        fig.add_trace(go.Scatter(
            x=predictions_df[date_column], 
            y=predictions_df[col],
            mode='lines', 
            name=col, 
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title=true_column,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_metrics(metrics_df, metrics_to_show=None, title="Performance Metrics per Model"):
    """
    Create a bar plot comparing metrics across models.
    
    Parameters:
        metrics_df (DataFrame): DataFrame from evaluate_models function
        metrics_to_show (list): List of metric names to include (default: all)
        title (str): Title for the plot
        
    Returns:
        Figure: Plotly figure object
    """
    if metrics_to_show is None:
        metrics_to_show = [col for col in metrics_df.columns if col != 'Model']
    
    fig = go.Figure()
    
    for metric in metrics_to_show:
        fig.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df[metric],
            name=metric
        ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Metric Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_error_distribution(errors_df, model_column, error_suffix='_error', bins=30):
    """
    Create a histogram of error distribution for a model.
    
    Parameters:
        errors_df (DataFrame): DataFrame from calculate_prediction_errors function
        model_column (str): Base name of the model column
        error_suffix (str): Suffix for the error column ('_error', '_abs_error', '_pct_error')
        bins (int): Number of bins for the histogram
        
    Returns:
        Figure: Plotly figure object
    """
    error_column = f"{model_column}{error_suffix}"
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors_df[error_column],
        nbinsx=bins,
        name=error_column
    ))
    
    fig.update_layout(
        title=f"Error Distribution for {model_column}",
        xaxis_title=error_column,
        yaxis_title="Frequency"
    )
    
    return fig

def plot_error_over_time(errors_df, date_column, model_columns, error_suffix='_error'):
    """
    Create a line plot of errors over time for multiple models.
    
    Parameters:
        errors_df (DataFrame): DataFrame from calculate_prediction_errors function
        date_column (str): Name of the date column
        model_columns (list): List of base model column names
        error_suffix (str): Suffix for the error column ('_error', '_abs_error', '_pct_error')
        
    Returns:
        Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for model in model_columns:
        error_column = f"{model}{error_suffix}"
        
        fig.add_trace(go.Scatter(
            x=errors_df[date_column],
            y=errors_df[error_column],
            mode='lines',
            name=model
        ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"Model Errors Over Time",
        xaxis_title=date_column,
        yaxis_title=f"Error{error_suffix}"
    )
    
    return fig

def plot_feature_importance(importances, feature_names, top_n=20, title=None):
    """
    Create a bar plot of feature importances.
    
    Parameters:
        importances (array): Array of feature importance values
        feature_names (array): Array of feature names
        top_n (int): Number of top features to show
        title (str): Custom title for the plot
        
    Returns:
        Figure: Plotly figure object
    """
    # Create a DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and select top N features
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h'
    ))
    
    if title is None:
        title = f"Top {top_n} Feature Importances"
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, 20 * top_n),  # Dynamic height based on the number of features
        yaxis=dict(autorange="reversed")  # Put most important feature at the top
    )
    
    return fig

def plot_prediction_scatter(predictions_df, true_column, prediction_columns):
    """
    Create scatter plots of predicted vs. true values for each model.
    
    Parameters:
        predictions_df (DataFrame): DataFrame containing true values and predictions
        true_column (str): Name of the column with true values
        prediction_columns (list): List of column names with model predictions
        
    Returns:
        Figure: Plotly figure object
    """
    # Create subplots based on the number of models
    n_cols = min(3, len(prediction_columns))
    n_rows = (len(prediction_columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=[f"Model: {col}" for col in prediction_columns])
    
    # Get the range of true values for consistent axes
    y_true = predictions_df[true_column]
    min_val = min(y_true.min(), min(predictions_df[col].min() for col in prediction_columns))
    max_val = max(y_true.max(), max(predictions_df[col].max() for col in prediction_columns))
    
    # Add a scatter plot for each model
    for i, col in enumerate(prediction_columns):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_true, 
                y=predictions_df[col],
                mode='markers',
                marker=dict(size=5),
                name=col
            ),
            row=row, col=col_idx
        )
        
        # Add perfect prediction line (y=x)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction' if i == 0 else None,
                showlegend=(i == 0)
            ),
            row=row, col=col_idx
        )
    
    # Update layout
    fig.update_layout(
        title="Predicted vs. True Values",
        height=300 * n_rows,
        width=300 * n_cols
    )
    
    # Update all xaxes and yaxes to have the same range
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            if (i-1) * n_cols + j <= len(prediction_columns):
                fig.update_xaxes(title_text="True Values", range=[min_val, max_val], row=i, col=j)
                fig.update_yaxes(title_text="Predicted Values", range=[min_val, max_val], row=i, col=j)
    
    return fig

def plot_residual_analysis(predictions_df, true_column, prediction_columns):
    """
    Create residual plots for each model.
    
    Parameters:
        predictions_df (DataFrame): DataFrame containing true values and predictions
        true_column (str): Name of the column with true values
        prediction_columns (list): List of column names with model predictions
        
    Returns:
        dict: Dictionary of Plotly figure objects for each model
    """
    figures = {}
    y_true = predictions_df[true_column]
    
    for col in prediction_columns:
        residuals = y_true - predictions_df[col]
        
        # Create a figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs. Fitted", "Residual Distribution")
        )
        
        # Residuals vs. fitted values
        fig.add_trace(
            go.Scatter(
                x=predictions_df[col],
                y=residuals,
                mode='markers',
                marker=dict(size=5),
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residual Distribution'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Residual Analysis for {col}",
            height=400,
            width=900
        )
        
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residual Value", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        figures[col] = fig
    
    return figures

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
        'model2': true_values + np.random.normal(0, 2, 100),
        'baseline': true_values + np.random.normal(0, 3, 100)
    })
    
    # Visualize predictions
    fig = visualize_predictions(predictions_df, 'date', 'actual', ['model1', 'model2', 'baseline'])
    fig.show()
    
    # Create sample metrics
    metrics = pd.DataFrame({
        'Model': ['model1', 'model2', 'baseline'],
        'MAE': [1.2, 2.1, 3.5],
        'RMSE': [1.5, 2.5, 4.0],
        'MAPE': [5.2, 8.3, 12.1]
    })
    
    # Plot metrics
    fig_metrics = plot_metrics(metrics)
    fig_metrics.show()
    
    # Create feature importances
    feature_names = ['Feature_' + str(i) for i in range(20)]
    importances = np.random.rand(20)
    
    # Plot feature importances
    fig = plot_feature_importance(importances, feature_names)
    fig.show()
