import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Get the current script directory and the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

# Import constants modules early to ensure they're in sys.modules
try:
    from constants.cases import CASE_DIC
    from constants.features import get_forecast_features, WEATHER_FEATURES
    print("Successfully imported constants modules")
except ImportError as e:
    print(f"Warning: Could not import constants modules: {e}")
    print("Prophet model will use fallback regressor columns")

# Now import modules using absolute imports
from models.regression_models import (
    LinearRegressionModel,
    LassoRegressionModel,
    RidgeRegressionModel,
    ElasticNetRegressionModel,
    LGBMModel,
    XGBoostModel,
    CatBoostModel,
    RandomForestModel,
    ProphetModel,
    BaselineModel,
    WeightedEnsembleModel
)

# Import hyperparameter tuning in a try-except block
try:
    from tuning.hyperparameter_tuning import OptunaHyperparameterTuner
    hyperparameter_tuning_available = True
except ImportError:
    print("Warning: OptunaHyperparameterTuner not available. "
          "Hyperparameter tuning will be disabled. "
          "Install optuna package to enable this feature.")
    hyperparameter_tuning_available = False

from evaluation.evaluation import evaluate_models
from evaluation.visualization import visualize_predictions, plot_feature_importance, plot_metrics

class TrainingPipeline:
    """Pipeline for training and evaluating regression models"""
    
    def __init__(self, data_path=None, case_id=None, target_column='production', 
                 date_column='effectivedate', test_size=0.2, random_state=42,
                 comparison_columns=None):
        """
        Initialize the training pipeline.
        
        Parameters:
            data_path (str): Path to the data file (if None, will use case_id to load from data/case_datasets)
            case_id (int): ID of the case to load from case_datasets
            target_column (str): Name of the target column
            date_column (str): Name of the date column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            comparison_columns (list): List of column names to include in comparison (e.g., provider forecasts)
        """
        self.target_column = target_column
        self.date_column = date_column
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
        self.plant_data = {}
        self.comparison_columns = comparison_columns or []
        self.case_id = case_id  # Store case_id for later use
        
        # Load data
        if data_path is not None:
            self.data = pd.read_csv(data_path)
        elif case_id is not None:
            # Find case file in data/case_datasets
            case_datasets_dir = os.path.join(project_root, 'data', 'case_datasets')
            case_files = list(Path(case_datasets_dir).glob(f'case_{case_id}_*.csv'))
            if not case_files:
                raise ValueError(f"No case file found for case_id {case_id}")
            self.data = pd.read_csv(case_files[0])
            print(f"Loaded case data from {case_files[0]}")
        else:
            raise ValueError("Either data_path or case_id must be provided")
        
        # Convert date column to datetime and sort
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data.sort_values(by=date_column, inplace=True)
        
        # Check if 'plant_name' column exists
        if 'plant_name' not in self.data.columns:
            raise ValueError("Data must include 'plant_name' column")
            
        # Validate comparison columns
        if self.comparison_columns:
            missing_cols = [col for col in self.comparison_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Comparison columns not found in data: {missing_cols}")
    
    def get_categorical_columns(self, df, threshold=20):
        """
        Identify categorical columns in a DataFrame.
        
        Parameters:
            df (DataFrame): DataFrame to analyze
            threshold (int): Maximum number of unique values for a numeric column to be considered categorical
            
        Returns:
            list: List of categorical column names
        """
        cat_cols = []
        for col in df.columns:
            # Skip target and date columns
            if col in [self.target_column, self.date_column]:
                continue
            
            # If already object or categorical, add it
            if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                cat_cols.append(col)
 
        return cat_cols
    
    def prepare_data_by_plant(self):
        """
        Split the data by plant and prepare train/test sets for each plant.
        
        Returns:
            dict: Dictionary of data splits by plant
        """
        plants = self.data['plant_name'].unique()
        print(f"Found {len(plants)} plants: {plants}")
        
        for plant in plants:
            plant_df = self.data[self.data['plant_name'] == plant].copy()
            X = plant_df.drop(columns=[self.target_column])
            y = plant_df[self.target_column]
            
            # Split data chronologically
            split_idx = int(len(plant_df) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Extract comparison values for test set if comparison columns are specified
            comparison_values = {}
            if self.comparison_columns:
                for col in self.comparison_columns:
                    comparison_values[col] = X_test[col].values if col in X_test.columns else None
            
            # Save split data
            self.plant_data[plant] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'categorical_features': self.get_categorical_columns(X_train),
                'comparison_values': comparison_values
            }
        
        return self.plant_data
    
    def prepare_features_for_regression(self, X_train, X_test):
        """
        Prepare features for regression models:
          - Drop date column
          - One-hot encode categorical features
          - Align train and test columns
        
        Parameters:
            X_train (DataFrame): Training features
            X_test (DataFrame): Test features
            
        Returns:
            tuple: (prepared X_train, prepared X_test)
        """
        # Make copies to avoid modifying the original data
        X_train_prep = X_train.drop(columns=[self.date_column])
        X_test_prep = X_test.drop(columns=[self.date_column])
        
        # One-hot encoding for categorical columns
        X_train_prep = pd.get_dummies(X_train_prep, drop_first=True)
        X_test_prep = pd.get_dummies(X_test_prep, drop_first=True)
        
        # Align columns between train and test (fill missing with 0)
        X_train_prep, X_test_prep = X_train_prep.align(X_test_prep, join='left', axis=1, fill_value=0)
        
        return X_train_prep, X_test_prep
    
    def prepare_data_for_prophet(self, train_df, test_df, regressor_columns=None):
        """
        Prepare data for Prophet model.
        
        Parameters:
            train_df (DataFrame): Training data
            test_df (DataFrame): Test data
            regressor_columns (list): List of regressor column names
            
        Returns:
            tuple: (prepared train_df, prepared test_df)
        """
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        if regressor_columns:
            for col in regressor_columns:
                if isinstance(train_df[col].dtype, pd.CategoricalDtype) or train_df[col].dtype == object:
                    train_df[col] = train_df[col].astype('category').cat.codes
                    test_df[col] = test_df[col].astype('category').cat.codes
        
        return train_df, test_df
    
    def train_models(self, models_to_train=None, use_hyperparameter_tuning=False, n_trials=50, baseline_shifts=None, ensemble_models=None):
        """
        Train models for each plant.
        
        Parameters:
            models_to_train (list): List of model names to train
                                   (default: ['linear', 'lasso', 'ridge', 'elastic_net', 
                                             'lgbm', 'xgboost', 'catboost', 'random_forest',
                                             'prophet', 'baseline'])
            use_hyperparameter_tuning (bool): Whether to use hyperparameter tuning
            n_trials (int): Number of trials for hyperparameter tuning
            baseline_shifts (list): List of shift values to use for baseline models (e.g., [1, 24, 168])
                                   If None, defaults to [24] (daily shift)
            ensemble_models (list): List of model names to include in weighted ensemble
                                    If None, no ensemble model will be created
            
        Returns:
            dict: Dictionary of trained models by plant
        """
        if models_to_train is None:
            models_to_train = ['linear', 'lasso', 'ridge', 'elastic_net', 
                              'lgbm', 'xgboost', 'catboost', 'random_forest',
                              'prophet', 'baseline']
        
        # Default baseline shifts if not specified
        if baseline_shifts is None:
            baseline_shifts = [24]  # Default to daily shift
        
        # Check if hyperparameter tuning is requested but not available
        if use_hyperparameter_tuning and not hyperparameter_tuning_available:
            print("Warning: Hyperparameter tuning was requested but is not available.")
            print("Install optuna package to enable this feature.")
            print("Proceeding with default hyperparameters.")
            use_hyperparameter_tuning = False
        
        if not self.plant_data:
            self.prepare_data_by_plant()
        
        results = {}
        
        for plant, data in self.plant_data.items():
            print(f"\nTraining models for plant {plant}...")
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            
            # Get capacity value for this plant (used for clipping predictions)
            capacity_value = X_train['capacity'].iloc[0]  # All values are the same
            
            # Create a single experiment_name for all model tuning
            experiment_name = f"{self.case_id}_plant_{plant}"
            
            # Drop capacity column since it's all the same value
            X_train = X_train.drop(columns=['capacity'])
            X_test = X_test.drop(columns=['capacity'])
            
            # Prepare data for regression models
            X_train_reg, X_test_reg = self.prepare_features_for_regression(X_train, X_test)
            
            # NOTE: We don't need to scale data here as all models handle scaling internally
            # Make dataframes from unscaled data to pass to models
            X_train_df = X_train_reg.copy()
            X_test_df = X_test_reg.copy()
            
            # Create validation set for ensemble weight optimization
            # Use a portion of the training set for validation (20% by default)
            val_size = 0.2
            val_split_idx = int(len(X_train_df) * (1 - val_size))
            X_val_df = X_train_df.iloc[val_split_idx:].copy()
            y_val = y_train.iloc[val_split_idx:].copy()
            
            # Adjust training set to exclude validation data
            X_train_df_no_val = X_train_df.iloc[:val_split_idx].copy()
            y_train_no_val = y_train.iloc[:val_split_idx].copy()
            
            # Create original validation features for Prophet and baseline
            X_val = X_train.iloc[val_split_idx:].copy()
            
            # Dictionary to store validation predictions from each model
            val_predictions = {}
            
            # Train models
            plant_models = {}
            predictions = {}
            
            for model_name in models_to_train:
                print(f"Training {model_name} model...")
                
                if model_name == 'linear':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('linear', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = LinearRegressionModel(normalize=True)
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'lasso':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('lasso', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = LassoRegressionModel(alpha=0.1)
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'ridge':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('ridge', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = RidgeRegressionModel(alpha=0.1)
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'elastic_net':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('elastic_net', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = ElasticNetRegressionModel(alpha=0.1, l1_ratio=0.5)
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'lgbm':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('lgbm', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = LGBMModel()
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'xgboost':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('xgboost', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = XGBoostModel()
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'catboost':
                    cat_features = data['categorical_features']
                    
                    # Filter cat_features to only include columns in X_train_reg
                    cat_features = [col for col in cat_features if col in X_train_reg.columns]
                    
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('catboost', n_trials=n_trials)
                        tuner.tune(X_train_reg.iloc[:val_split_idx], y_train_no_val, 
                                   categorical_features=cat_features, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = CatBoostModel()
                    
                    # First train on the training portion
                    model.fit(X_train_reg.iloc[:val_split_idx], y_train_no_val, cat_features=cat_features)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_train_reg.iloc[val_split_idx:])
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_reg, y_train, cat_features=cat_features)
                    preds = model.predict(X_test_reg)
                
                elif model_name == 'random_forest':
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('random_forest', n_trials=n_trials)
                        tuner.tune(X_train_df_no_val, y_train_no_val, experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = RandomForestModel()
                    
                    # First train on the training portion
                    model.fit(X_train_df_no_val, y_train_no_val)
                    
                    # Get validation predictions for ensemble weight optimization
                    val_predictions[model_name] = model.predict(X_val_df)
                    
                    # Now train on the full dataset for test predictions
                    model.fit(X_train_df, y_train)
                    preds = model.predict(X_test_df)
                
                elif model_name == 'prophet':
                    # Get the case id from the data path or use a default
                    case_id = None
                    if hasattr(self, 'case_id'):
                        case_id = self.case_id
                        print(f"DEBUG: Found case_id {case_id} from class attribute")
                    else:
                        # Try to extract case_id from data path if available
                        try:
                            data_paths = list(Path(os.path.join(project_root, 'data', 'case_datasets')).glob('case_*_*.csv'))
                            if data_paths:
                                case_id = int(str(data_paths[0]).split('_')[1])
                                print(f"DEBUG: Extracted case_id {case_id} from data path: {data_paths[0]}")
                            else:
                                print(f"DEBUG: No case files found in {os.path.join(project_root, 'data', 'case_datasets')}")
                        except Exception as e:
                            print(f"DEBUG: Error extracting case_id: {e}")
                    
                    # Get forecast features for this case if possible
                    print(f"DEBUG: case_id = {case_id}, 'constants.cases' in sys.modules = {'constants.cases' in sys.modules}")
                    if case_id is not None:
                        try:
                            # Store the original in __init__ to make it available here
                            if not hasattr(self, 'case_id'):
                                self.case_id = case_id
                                
                            from constants.cases import CASE_DIC
                            from constants.features import get_forecast_features
                            
                            if case_id in CASE_DIC:
                                # Get provider numbers from the case data
                                providers = [int(p.split()[1]) for p in CASE_DIC[case_id]["providers"]]
                                # Get forecast type - either 'Last FC' or 'Dayahead FC'
                                is_dayahead = CASE_DIC[case_id]["type"] == "Dayahead FC"
                                
                                print(f"DEBUG: Found case {case_id} with providers {providers}, dayahead={is_dayahead}")
                                
                                # Get forecast features for this case
                                regressor_columns = get_forecast_features(providers, da=is_dayahead)
                                
                                # Filter to ensure all columns exist in the data
                                original_count = len(regressor_columns)
                                regressor_columns = [col for col in regressor_columns if col in X_train.columns]
                                
                                print(f"DEBUG: Using {len(regressor_columns)}/{original_count} forecast features as regressors for Prophet model")
                                if len(regressor_columns) < original_count:
                                    print(f"DEBUG: Missing columns: {set(get_forecast_features(providers, da=is_dayahead)) - set(regressor_columns)}")
                            else:
                                print(f"DEBUG: Case ID {case_id} not found in CASE_DIC")
                                # Default fallback - use all non-date, non-plant columns
                                regressor_columns = [col for col in X_train.columns 
                                                   if col not in [self.date_column, 'plant_name']]
                                print(f"DEBUG: Using {len(regressor_columns)} columns as fallback")
                        except ImportError as e:
                            print(f"DEBUG: ImportError when importing constants: {e}")
                            # Default fallback if constants.cases isn't available
                            regressor_columns = [col for col in X_train.columns 
                                               if col not in [self.date_column, 'plant_name']]
                            print(f"DEBUG: Using {len(regressor_columns)} columns as fallback after ImportError")
                    else:
                        print("DEBUG: No case_id available, using fallback")
                        # Default fallback - use all non-date, non-plant columns
                        regressor_columns = [col for col in X_train.columns 
                                           if col not in [self.date_column, 'plant_name']]
                        print(f"DEBUG: Using {len(regressor_columns)} columns as fallback")
                    
                    # Prepare data for Prophet - first for training and validation
                    train_df_prophet_no_val = X_train.iloc[:val_split_idx].reset_index(drop=True).copy()
                    train_df_prophet_no_val[self.target_column] = y_train_no_val.reset_index(drop=True)
                    
                    val_df_prophet = X_val.reset_index(drop=True).copy()  # For validation predictions
                    
                    # Prepare data for full training and test predictions
                    train_df_prophet_full = X_train.reset_index(drop=True).copy()
                    train_df_prophet_full[self.target_column] = y_train.reset_index(drop=True)
                    
                    test_df_prophet = X_test.reset_index(drop=True).copy()
                    
                    if use_hyperparameter_tuning and hyperparameter_tuning_available:
                        tuner = OptunaHyperparameterTuner('prophet', n_trials=n_trials)
                        tuner.tune(train_df_prophet_no_val, date_column=self.date_column, 
                                  target_column=self.target_column, regressor_columns=regressor_columns,
                                  experiment_name=experiment_name)
                        model = tuner.get_best_model()
                    else:
                        model = ProphetModel()
                    
                    # First train on the training portion only
                    model.fit(train_df_prophet_no_val, self.date_column, self.target_column, regressor_columns)
                    
                    # Get validation predictions for ensemble weight optimization
                    try:
                        val_predictions[model_name] = model.predict(val_df_prophet, self.date_column, regressor_columns)
                    except Exception as e:
                        print(f"Warning: Could not get Prophet validation predictions: {e}")
                        # Don't include in ensemble if validation predictions fail
                    
                    # Now train on the full dataset for test predictions
                    try:
                        model.fit(train_df_prophet_full, self.date_column, self.target_column, regressor_columns)
                        preds = model.predict(test_df_prophet, self.date_column, regressor_columns)
                    except Exception as e:
                        print(f"Warning: Prophet model failed on test predictions: {e}")
                        print("Using validation model for test predictions")
                        # Use the validation model for predictions if the full model fails
                        preds = model.predict(test_df_prophet, self.date_column, regressor_columns)
                
                elif model_name == 'baseline':
                    # Create baseline models with the specified shift values
                    baselines = {}
                    for shift in baseline_shifts:
                        baselines[f'baseline_shift{shift}'] = BaselineModel(shift=shift)
                    
                    # Train and predict with each baseline model
                    for baseline_name, baseline_model in baselines.items():
                        try:
                            # For validation predictions, use only the training portion + validation portion
                            train_y_series_no_val = pd.Series(y_train_no_val, name=self.target_column)
                            val_y_series = pd.Series(y_val, name=self.target_column)
                            val_data = pd.concat([train_y_series_no_val, val_y_series]).reset_index(drop=True)
                            
                            # Fit on validation data to get validation predictions
                            baseline_model.fit(pd.DataFrame({self.target_column: val_data}), self.target_column)
                            val_preds = baseline_model.predict(len(y_val))
                            val_predictions[baseline_name] = val_preds
                            
                            # Create a combined dataset with both train and test data in order
                            # This is needed for the baseline model to correctly access shifted values
                            train_y_series = pd.Series(y_train, name=self.target_column)
                            test_y_series = pd.Series(y_test, name=self.target_column)
                            all_y = pd.concat([train_y_series, test_y_series]).reset_index(drop=True)
                            
                            # Train the baseline model on the whole series
                            baseline_model.fit(pd.DataFrame({self.target_column: all_y}), self.target_column)
                            
                            # Ask it to predict the test portion only
                            baseline_preds = baseline_model.predict(len(y_test))
                            
                            plant_models[baseline_name] = baseline_model
                            predictions[baseline_name] = baseline_preds
                        except Exception as e:
                            print(f"Warning: Failed to train {baseline_name} model: {e}")
                    
                    # No need to return anything here since we've already added the predictions to the dictionary
                    continue
                
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
                
                # Store model and predictions
                plant_models[model_name] = model
                
                # Clip predictions to be between 0 and the plant's capacity
                preds = np.clip(preds, 0, capacity_value)
                
                predictions[model_name] = preds
            
            # Create ensemble model if requested
            if ensemble_models is not None and len(ensemble_models) > 1:
                print(f"Creating weighted ensemble model with {len(ensemble_models)} base models: {ensemble_models}")
                
                # Check that all requested models were trained and have predictions
                valid_ensemble_models = []
                for model_name in ensemble_models:
                    if model_name in predictions and model_name in val_predictions:
                        valid_ensemble_models.append(model_name)
                    else:
                        print(f"Warning: Model '{model_name}' not found in predictions or validation predictions. Skipping in ensemble.")
                
                if len(valid_ensemble_models) > 1:
                    print(f"Using {len(valid_ensemble_models)} models for ensemble: {valid_ensemble_models}")
                    
                    # Initialize optimized_weights with equal weights as fallback
                    optimized_weights = {name: 1.0/len(valid_ensemble_models) for name in valid_ensemble_models}
                    
                    # Optimize weights using the pre-computed validation predictions
                    print(f"Optimizing ensemble weights using validation set predictions...")
                    
                    try:
                        # Use Optuna for weight optimization
                        import optuna
                        # Import metrics needed for calculations
                        from sklearn.metrics import mean_squared_error, mean_absolute_error
                        
                        # Use the same metric calculation functions as in the hyperparameter tuner
                        def _calculate_mape(y_true, y_pred):
                            """Calculate Mean Absolute Percentage Error (MAPE)"""
                            # Filter out zeros in y_true to avoid division by zero
                            mask = y_true != 0
                            if not np.any(mask):
                                return float('inf')  # Return infinity if all true values are zero
                            
                            y_true_filtered = y_true[mask]
                            y_pred_filtered = y_pred[mask]
                            
                            # Calculate MAPE
                            mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
                            return mape
                        
                        def _calculate_weighted_score(y_true, y_pred):
                            """Calculate weighted score combining RMSE, MAE, and MAPE"""
                            # Calculate individual metrics
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)
                            mape = _calculate_mape(y_true, y_pred)
                            
                            # Normalize metrics (optional, based on typical ranges)
                            rmse_normalized = rmse / 100  # Assuming RMSE typically in 0-100 range
                            mae_normalized = mae / 50     # Assuming MAE typically in 0-50 range
                            mape_normalized = mape / 50   # Assuming MAPE typically in 0-50% range
                            
                            # Combine metrics with specific weights
                            # Using the same weights as in hyperparameter tuning: rmse=0.4, mae=0.3, mape=0.3
                            weighted_score = (
                                0.4 * rmse_normalized +
                                0.3 * mae_normalized +
                                0.3 * mape_normalized
                            )
                            
                            return weighted_score
                        
                        def objective(trial):
                            # Get weights using Optuna
                            raw_weights = {}
                            for model_name in valid_ensemble_models:
                                raw_weights[model_name] = trial.suggest_float(
                                    f"weight_{model_name}", 0.1, 10.0, log=True
                                )
                                
                            # Normalize weights to sum to 1.0
                            weight_sum = sum(raw_weights.values())
                            normalized_weights = {
                                model_name: raw_weight / weight_sum 
                                for model_name, raw_weight in raw_weights.items()
                            }
                            
                            # Apply min_weight constraint (0.05 minimum weight)
                            min_weight = 0.05
                            below_min = [name for name, w in normalized_weights.items() if w < min_weight]
                            if below_min:
                                min_total = len(below_min) * min_weight
                                remaining = 1.0 - min_total
                                above_min = [name for name in valid_ensemble_models if name not in below_min]
                                
                                if not above_min:
                                    # If all models would be below min, use equal weights
                                    normalized_weights = {name: 1.0/len(valid_ensemble_models) for name in valid_ensemble_models}
                                else:
                                    above_min_sum = sum(raw_weights[name] for name in above_min)
                                    for name in valid_ensemble_models:
                                        if name in below_min:
                                            normalized_weights[name] = min_weight
                                        else:
                                            normalized_weights[name] = (
                                                min_weight + 
                                                (raw_weights[name] / above_min_sum) * 
                                                (remaining - len(above_min) * min_weight)
                                            )
                            
                            # Ensure weights sum to 1.0
                            w_sum = sum(normalized_weights.values())
                            if not np.isclose(w_sum, 1.0, atol=1e-10):
                                for name in normalized_weights:
                                    normalized_weights[name] /= w_sum
                            
                            # Calculate weighted ensemble predictions for validation set
                            weighted_pred = np.zeros(len(y_val))
                            for model_name, weight in normalized_weights.items():
                                weighted_pred += weight * val_predictions[model_name]
                            
                            # Use the same weighted metric as in hyperparameter tuning
                            weighted_score = _calculate_weighted_score(y_val, weighted_pred)
                            
                            return weighted_score  # Lower is better
                        
                        # Run optimization
                        study = optuna.create_study(direction="minimize")
                        study.optimize(objective, n_trials=n_trials)
                        
                        # Get best weights
                        best_params = study.best_params
                        raw_weights = {}
                        
                        for param, value in best_params.items():
                            if param.startswith("weight_"):
                                model_name = param[len("weight_"):]
                                raw_weights[model_name] = value
                        
                        # Normalize and apply min_weight
                        weight_sum = sum(raw_weights.values())
                        optimized_weights = {
                            model_name: raw_weight / weight_sum 
                            for model_name, raw_weight in raw_weights.items()
                        }
                        
                        # Apply min_weight constraint
                        min_weight = 0.05
                        below_min = [name for name, w in optimized_weights.items() if w < min_weight]
                        if below_min:
                            min_total = len(below_min) * min_weight
                            remaining = 1.0 - min_total
                            above_min = [name for name in valid_ensemble_models if name not in below_min]
                            
                            if above_min:
                                above_min_sum = sum(raw_weights[name] for name in above_min)
                                for name in optimized_weights:
                                    if name in below_min:
                                        optimized_weights[name] = min_weight
                                    else:
                                        proportion = raw_weights[name] / above_min_sum
                                        optimized_weights[name] = (
                                            min_weight + 
                                            proportion * (remaining - len(above_min) * min_weight)
                                        )
                            else:
                                # If all models would be below min, use equal weights
                                equal_weight = 1.0 / len(valid_ensemble_models)
                                optimized_weights = {name: equal_weight for name in valid_ensemble_models}
                        
                        # Final normalization
                        w_sum = sum(optimized_weights.values())
                        for name in optimized_weights:
                            optimized_weights[name] /= w_sum
                        
                        # Calculate final validation score with these weights
                        final_val_pred = np.zeros(len(y_val))
                        for model_name, weight in optimized_weights.items():
                            final_val_pred += weight * val_predictions[model_name]
                        
                        final_weighted_score = _calculate_weighted_score(y_val, final_val_pred)
                        final_rmse = np.sqrt(mean_squared_error(y_val, final_val_pred))
                        final_mae = mean_absolute_error(y_val, final_val_pred)
                        final_mape = _calculate_mape(y_val, final_val_pred)
                        
                        print(f"Optimized weights: {optimized_weights}")
                        print(f"Best weighted score: {study.best_value:.6f}")
                        print(f"Final validation metrics - RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}, MAPE: {final_mape:.4f}%")
                        
                    except Exception as e:
                        print(f"Error optimizing weights: {e}")
                        print("Using equal weights...")
                        # optimized_weights is already initialized with equal weights before the try block
                    
                    # Create ensemble prediction using TEST predictions directly
                    ensemble_preds = np.zeros(len(y_test))
                    for model_name, weight in optimized_weights.items():
                        ensemble_preds += weight * predictions[model_name]
                    
                    # Clip predictions
                    ensemble_preds = np.clip(ensemble_preds, 0, capacity_value)
                    
                    # Add ensemble predictions to results
                    predictions['weighted_ensemble'] = ensemble_preds
                    
                    # Store ensemble information
                    ensemble_info = {
                        'weights': optimized_weights,
                        'base_models': valid_ensemble_models
                    }
                    plant_models['weighted_ensemble'] = ensemble_info
                    
                    print(f"Ensemble model added with {len(valid_ensemble_models)} base models.")
                    print(f"Final ensemble weights: {optimized_weights}")
                else:
                    print("Not enough valid models for ensemble. Skipping ensemble creation.")
            
            # For baseline models, also clip predictions
            if 'baseline' in models_to_train:
                for baseline_name in [name for name in predictions.keys() if 'baseline' in name]:
                    predictions[baseline_name] = np.clip(predictions[baseline_name], 0, capacity_value)
            
            # Store results for this plant
            comparison_values_clipped = {}
            for col, values in data['comparison_values'].items():
                if values is not None:
                    comparison_values_clipped[col] = np.clip(values, 0, capacity_value)
                else:
                    comparison_values_clipped[col] = values
            
            results[plant] = {
                'models': plant_models,
                'predictions': predictions,
                'y_test': y_test,
                'test_dates': X_test[self.date_column],
                'comparison_values': comparison_values_clipped,
                'capacity': capacity_value  # Store capacity value for reference
            }
        
        self.models = results
        return results
    
    def evaluate(self, output_dir='results'):
        """
        Evaluate trained models and save results.
        
        All predictions are already clipped between 0 and plant capacity during training.
        
        Parameters:
            output_dir (str): Directory to save evaluation results
            
        Returns:
            dict: Dictionary of evaluation metrics by plant
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        evaluation_results = {}
        
        # Check if this case has weather features
        has_weather_features = False
        try:
            if self.case_id is not None:
                from constants.cases import CASE_DIC
                from constants.features import WEATHER_FEATURES
                
                if self.case_id in CASE_DIC:
                    # Check if "features" key in the case dictionary contains any weather features
                    case_features = CASE_DIC[self.case_id]["features"]
                    has_weather_features = any(feature in WEATHER_FEATURES for feature in case_features)
                    print(f"Case {self.case_id} {'has' if has_weather_features else 'does not have'} weather features")
            else:
                # Check if any weather features exist in our dataset
                for plant_data in self.plant_data.values():
                    X_train = plant_data['X_train']
                    has_weather_features = any(feature in X_train.columns for feature in WEATHER_FEATURES)
                    if has_weather_features:
                        break
                print(f"Dataset {'has' if has_weather_features else 'does not have'} weather features")
        except (ImportError, KeyError) as e:
            print(f"Warning: Could not determine if case has weather features: {e}")
            # Conservative approach - assume it might have weather features
            has_weather_features = True
        
        for plant, result in self.models.items():
            predictions = result['predictions']
            y_test = result['y_test']
            test_dates = result['test_dates']
            comparison_values = result.get('comparison_values', {})
            
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                self.date_column: test_dates,
                'actual': y_test
            })
            
            # Add model predictions
            for model_name, preds in predictions.items():
                pred_df[model_name] = preds
                
            # Add comparison values
            for col, values in comparison_values.items():
                if values is not None:
                    pred_df[col] = values
            
            # Create list of columns to evaluate - model predictions + comparison columns
            eval_columns = list(predictions.keys()) + list(comparison_values.keys())
            
            # Evaluate models
            metrics = evaluate_models(pred_df, 'actual', eval_columns, self.date_column)
            
            # Save predictions
            pred_df.to_csv(f"{output_dir}/predictions_plant_{plant}.csv", index=False)
            
            # Save metrics
            metrics.to_csv(f"{output_dir}/metrics_plant_{plant}.csv", index=False)
            
            # Visualize all predictions in a single plot (including comparison columns)
            fig = visualize_predictions(
                pred_df, 
                self.date_column, 
                'actual', 
                eval_columns,
                title=f"All Predictions for Plant {plant}"
            )
            fig.write_html(f"{output_dir}/predictions_plot_plant_{plant}.html")
            
            # Create bar plot of metrics
            metrics_fig = plot_metrics(
                metrics, 
                title=f"Performance Metrics for Plant {plant}"
            )
            metrics_fig.write_html(f"{output_dir}/metrics_plot_plant_{plant}.html")
            
            # Plot feature importance for tree-based models
            for model_name in ['random_forest', 'lgbm', 'xgboost', 'catboost']:
                if model_name in result['models']:
                    if hasattr(result['models'][model_name], 'feature_importance'):
                        try:
                            # Get feature importance values
                            importances = result['models'][model_name].feature_importance()
                            
                            # Get the feature names from the same data used for training
                            # Use X_train_df since that's what was used for training these models
                            X_train_df = self.prepare_features_for_regression(
                                self.plant_data[plant]['X_train'].drop(columns=['capacity']), 
                                self.plant_data[plant]['X_test'].drop(columns=['capacity'])
                            )[0]
                            
                            # Ensure lengths match
                            if len(importances) == len(X_train_df.columns):
                                # Standard feature importance plot
                                fig = plot_feature_importance(importances, X_train_df.columns)
                                fig.write_html(f"{output_dir}/feature_importance_{model_name}_plant_{plant}.html")
                                
                                # Create weather-specific feature importance plot if weather features exist
                                if has_weather_features:
                                    # Create a mask for weather features
                                    try:
                                        from constants.features import WEATHER_FEATURES
                                        
                                        # Get weather features that actually exist in our data
                                        # Note: Due to one-hot encoding, we need to check if any column contains a weather feature name
                                        weather_cols = []
                                        for col in X_train_df.columns:
                                            # Check if the column name contains any weather feature
                                            for wf in WEATHER_FEATURES:
                                                if wf in col:
                                                    weather_cols.append(col)
                                                    break
                                        
                                        if weather_cols:
                                            # Create a mask for weather features
                                            weather_mask = [col in weather_cols for col in X_train_df.columns]
                                            
                                            # Filter importances and column names
                                            weather_importances = importances[weather_mask]
                                            weather_cols_filtered = X_train_df.columns[weather_mask]
                                            
                                            if len(weather_importances) > 0:
                                                # Create weather-specific plot
                                                weather_fig = plot_feature_importance(
                                                    weather_importances, 
                                                    weather_cols_filtered,
                                                    title=f"Weather Feature Importance for {model_name.capitalize()} - Plant {plant}"
                                                )
                                                weather_fig.write_html(f"{output_dir}/weather_feature_importance_{model_name}_plant_{plant}.html")
                                                print(f"Created weather feature importance plot for {model_name} - Plant {plant}")
                                            else:
                                                print(f"No weather features found for {model_name} - Plant {plant}")
                                        else:
                                            print(f"No weather features detected in the data for Plant {plant}")
                                    except (ImportError, Exception) as e:
                                        print(f"Warning: Could not create weather feature importance plot: {e}")
                            else:
                                print(f"Warning: Feature importance length mismatch for {model_name}: {len(importances)} vs {len(X_train_df.columns)} columns")
                                # Try using the first N features that match the importances length
                                if len(importances) < len(X_train_df.columns):
                                    print(f"Using the first {len(importances)} features for plotting")
                                    fig = plot_feature_importance(importances, X_train_df.columns[:len(importances)])
                                    fig.write_html(f"{output_dir}/feature_importance_{model_name}_plant_{plant}.html")
                        except (AttributeError, ValueError) as e:
                            print(f"Could not plot feature importance for {model_name}: {e}")
            
            evaluation_results[plant] = metrics
        
        return evaluation_results
    
    def get_aggregate_results(self, output_dir='results'):
        """
        Aggregate results across all plants and evaluate.
        
        Parameters:
            output_dir (str): Directory to save evaluation results
            
        Returns:
            tuple: (aggregated DataFrame with predictions, metrics DataFrame)
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all predictions
        all_predictions = []
        
        for plant, result in self.models.items():
            predictions = result['predictions']
            y_test = result['y_test']
            test_dates = result['test_dates']
            comparison_values = result.get('comparison_values', {})
            
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                self.date_column: test_dates,
                'plant_name': plant,
                'actual': y_test
            })
            
            # Add model predictions
            for model_name, preds in predictions.items():
                pred_df[model_name] = preds
                
            # Add comparison values
            for col, values in comparison_values.items():
                if values is not None:
                    pred_df[col] = values
            
            all_predictions.append(pred_df)
        
        # Combine all predictions
        all_pred_df = pd.concat(all_predictions)
        
        # Aggregate by date
        agg_pred_df = all_pred_df.groupby(self.date_column).sum().reset_index()
        
        # Remove 'plant_name' from the columns to aggregate
        if 'plant_name' in agg_pred_df.columns:
            agg_pred_df = agg_pred_df.drop('plant_name', axis=1)
        
        # Create list of all columns to evaluate
        model_names = [col for col in agg_pred_df.columns 
                      if col not in [self.date_column, 'actual']]
        
        # Evaluate aggregated predictions
        metrics = evaluate_models(agg_pred_df, 'actual', model_names, self.date_column)
        
        # Save results
        agg_pred_df.to_csv(f"{output_dir}/predictions_aggregate.csv", index=False)
        metrics.to_csv(f"{output_dir}/metrics_aggregate.csv", index=False)
        
        # Visualize all predictions in a single plot
        fig = visualize_predictions(
            agg_pred_df, 
            self.date_column, 
            'actual', 
            model_names,
            title="All Aggregated Predictions"
        )
        fig.write_html(f"{output_dir}/predictions_plot_aggregate.html")
        
        # Create bar plot of metrics
        metrics_fig = plot_metrics(
            metrics, 
            title="Performance Metrics (Aggregated)"
        )
        metrics_fig.write_html(f"{output_dir}/metrics_plot_aggregate.html")
        
        return agg_pred_df, metrics

# Example usage
if __name__ == "__main__":
    # Train models for case 1
    pipeline = TrainingPipeline(case_id=1)
    pipeline.train_models(models_to_train=['lgbm', 'xgboost', 'catboost', 'prophet', 'baseline'])
    pipeline.evaluate()
    pipeline.get_aggregate_results()
