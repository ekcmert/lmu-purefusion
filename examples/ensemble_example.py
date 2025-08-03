import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from train.training_pipeline import TrainingPipeline
import pandas as pd
import numpy as np
import time

if __name__ == "__main__":
    # Specify a case ID to use
    case_id = 1  # Change this to use a different case
    
    # Specify which models to use
    models_to_train = ['linear', 'lasso', 'ridge', 'elastic_net', 'lgbm', 'xgboost', 'random_forest']
    
    # Specify which models to include in the ensemble
    ensemble_models = ['ridge', 'elastic_net', 'lgbm', 'xgboost']
    
    print(f"Starting example with case {case_id}...")
    print(f"Training models: {models_to_train}")
    print(f"Ensemble models: {ensemble_models}")
    
    # Create results directory
    results_dir = os.path.join(project_root, "results", "ensemble_example")
    os.makedirs(results_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        case_id=case_id,
        test_size=0.2,
        random_state=42
    )
    
    # Train models with ensemble
    pipeline.train_models(
        models_to_train=models_to_train,
        use_hyperparameter_tuning=True,
        n_trials=10,  # Use a small number of trials for faster execution
        ensemble_models=ensemble_models
    )
    
    # Evaluate models
    pipeline.evaluate(output_dir=results_dir)
    
    # Get aggregate results
    agg_preds, agg_metrics = pipeline.get_aggregate_results(output_dir=results_dir)
    
    # Print metrics
    print("\nAggregate metrics:")
    for index, row in agg_metrics.iterrows():
        print(f"{row['Model']}: RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")
    
    # Record end time and print duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nExecution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    print(f"\nResults saved to {results_dir}") 