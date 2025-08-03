import os
import sys

# Add the project root to the path so we can import modules
# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)
# Add the project root to sys.path
sys.path.append(parent_dir)

from train.training_pipeline import TrainingPipeline
from constants.cases import CASE_DIC
import time

def get_comparison_columns(case_id):
    """
    Get comparison columns for a specific case based on its type.
    
    Args:
        case_id (int): Case ID
    
    Returns:
        list: List of comparison columns
    """
    if case_id not in CASE_DIC:
        raise ValueError(f"Case ID {case_id} not found in CASE_DIC")
    
    case_info = CASE_DIC[case_id]
    case_type = case_info["type"]
    providers = case_info["providers"]
    
    # Convert provider names to provider numbers
    provider_numbers = [int(provider.split(" ")[1]) for provider in providers]
    
    # Set appropriate columns based on case type
    if case_type == "Dayahead FC":
        # For dayahead cases, use fc1200 columns
        return [f"provider{num}_fc1200" for num in provider_numbers]
    elif case_type == "Last FC":
        # For last FC cases, use fc0 columns
        return [f"provider{num}_fc0" for num in provider_numbers]
    else:
        raise ValueError(f"Unknown case type: {case_type}")

if __name__ == "__main__":
    # Train models for all cases
    print("Starting training pipeline for all cases...")
    
    # Create a results directory if it doesn't exist
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define models to train
    all_models = ['linear', 'lasso', 'ridge', 'elastic_net', 'lgbm', 'xgboost', 'catboost', 'random_forest', 'prophet', 'baseline']

    # Define which models to include in the ensemble (if any)
    ensemble_models = ['elastic_net', 'lgbm', 'xgboost', 'catboost', 'random_forest', 'prophet']
    
    # Define hyperparameter tuning settings
    use_tuning = True
    n_trials = 40
    test_size = 0.2
    random_state = 42
    
    # Loop through all case IDs
    # for case_id in CASE_DIC.keys():
    for case_id in [3, 4, 5, 6, 7]:
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Processing Case {case_id} - {CASE_DIC[case_id]['type']}")
        print(f"{'='*50}")
        
        # Get appropriate comparison columns for this case
        try:
            comparison_columns = get_comparison_columns(case_id)
            print(f"Using comparison columns: {comparison_columns}")
        except Exception as e:
            print(f"Error getting comparison columns: {e}")
            print("Continuing without comparison columns.")
            comparison_columns = None
        
        # Create output directory for this case
        case_dir = os.path.join(results_dir, f'case_{case_id}_{CASE_DIC[case_id]["type"].lower().replace(" ", "_")}')
        if use_tuning:
            case_dir = f"{case_dir}_tuned_{n_trials}trials"
        os.makedirs(case_dir, exist_ok=True)
        
        # Initialize the pipeline
        print(f"\nInitializing pipeline with case_id={case_id} and comparison columns...")
        try:
            pipeline = TrainingPipeline(
                case_id=case_id,
                comparison_columns=comparison_columns,
                test_size=test_size,
                random_state=random_state
            )
            
            # Train the models
            print(f"\nTraining models for case {case_id} with hyperparameter tuning={use_tuning}...")
            pipeline.train_models(
                models_to_train=all_models,
                use_hyperparameter_tuning=use_tuning,
                n_trials=n_trials,
                baseline_shifts=[1],  # Using a 1-hour shift for baseline model
                ensemble_models=ensemble_models  # Use ensemble models
            )
            
            # Evaluate models and compare against provider forecasts
            print(f"\nEvaluating models for case {case_id}...")
            plant_metrics = pipeline.evaluate(output_dir=case_dir)
            
            # Get aggregate results with comparison
            print(f"\nCalculating aggregate results for case {case_id}...")
            agg_preds, agg_metrics = pipeline.get_aggregate_results(output_dir=case_dir)
            
            # Print the aggregated metrics to show comparison
            print(f"\nAggregate metrics for case {case_id} (including provider forecasts):")
            print(agg_metrics[["Model", "MAE", "RMSE"]])
            
        except Exception as e:
            print(f"Error during processing of case {case_id}: {e}")
            
            # Try again without hyperparameter tuning if it was enabled
            if use_tuning:
                print(f"\nRetrying case {case_id} without hyperparameter tuning...")
                fallback_dir = f"{case_dir.replace('_tuned_', '_no_tuning_')}"
                os.makedirs(fallback_dir, exist_ok=True)
                
                try:
                    # Reinitialize the pipeline
                    pipeline = TrainingPipeline(
                        case_id=case_id,
                        comparison_columns=comparison_columns,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Train without hyperparameter tuning
                    pipeline.train_models(
                        models_to_train=all_models,
                        use_hyperparameter_tuning=False,
                        baseline_shifts=[1],
                        ensemble_models=ensemble_models  # Use ensemble models
                    )
                    
                    # Evaluate models
                    plant_metrics = pipeline.evaluate(output_dir=fallback_dir)
                    
                    # Get aggregate results
                    agg_preds, agg_metrics = pipeline.get_aggregate_results(output_dir=fallback_dir)
                    
                    print(f"\nAggregate metrics for case {case_id} (no hyperparameter tuning):")
                    print(agg_metrics[["Model", "MAE", "RMSE"]])
                    
                except Exception as inner_e:
                    print(f"Error during fallback processing of case {case_id}: {inner_e}")
        
        # Calculate and print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time for case {case_id}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print("\nAll cases completed!") 