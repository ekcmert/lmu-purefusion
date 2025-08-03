import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import sys
import os
sys.path.append(os.path.abspath('..'))
from models.regression_models import (
    LinearRegressionModel,
    LassoRegressionModel,
    RidgeRegressionModel,
    ElasticNetRegressionModel,
    LGBMModel,
    XGBoostModel,
    CatBoostModel,
    RandomForestModel,
    ProphetModel
)

class OptunaHyperparameterTuner:
    """Class for hyperparameter tuning using Optuna"""
    
    def __init__(self, model_type, metric='weighted', n_trials=100, validation_size=0.2, random_state=42,
                metric_weights=None):
        """
        Initialize the hyperparameter tuner.
        
        Parameters:
            model_type (str): Type of model to tune ('linear', 'lasso', 'ridge', 'elastic_net', 
                             'lgbm', 'xgboost', 'catboost', 'random_forest', 'prophet')
            metric (str): Metric to optimize ('weighted', 'rmse', 'mae', 'r2', 'mape')
            n_trials (int): Number of Optuna trials
            validation_size (float): Proportion of training data to use for validation (0.0-1.0)
            random_state (int): Random seed for reproducibility
            metric_weights (dict): Weights for each metric when using 'weighted' metric
                                   Default: {'rmse': 0.4, 'mae': 0.3, 'mape': 0.3}
        """
        self.model_type = model_type
        self.metric = metric
        self.n_trials = n_trials
        self.validation_size = validation_size
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None
        
        # Set default metric weights if not provided
        if metric_weights is None and metric == 'weighted':
            self.metric_weights = {'rmse': 0.4, 'mae': 0.3, 'mape': 0.3}
        else:
            self.metric_weights = metric_weights
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_time_split(self, X, y=None):
        """
        Create a single time-based train/validation split for time series data.
        
        Parameters:
            X (DataFrame): Feature DataFrame
            y (Series): Target Series
            
        Returns:
            tuple: (train_X, train_y, val_X, val_y)
        """
        # Calculate the validation split point
        total_samples = len(X)
        val_start = int(total_samples * (1 - self.validation_size))
        
        # Create train/validation split - data is already sorted by date
        train_X = X.iloc[:val_start].copy()
        val_X = X.iloc[val_start:].copy()
        
        if y is not None:
            train_y = y.iloc[:val_start].copy()
            val_y = y.iloc[val_start:].copy()
            return train_X, train_y, val_X, val_y
        else:
            return train_X, None, val_X, None
    
    def _get_objective_func(self, X_train, y_train=None, categorical_features=None,
                         date_column=None, target_column=None, regressor_columns=None):
        """
        Return the appropriate objective function based on model type.
        
        Parameters:
            X_train (array-like): Training features or full DataFrame for time series models
            y_train (array-like): Training targets (not needed for Prophet)
            categorical_features (list): List of categorical feature indices or names (for CatBoost)
            date_column (str): Date column name (for Prophet)
            target_column (str): Target column name (for Prophet)
            regressor_columns (list): List of regressor column names (for Prophet)
            
        Returns:
            function: Objective function for Optuna
        """
        if self.model_type == 'linear':
            return self._objective_linear(X_train, y_train)
        elif self.model_type == 'lasso':
            return self._objective_lasso(X_train, y_train)
        elif self.model_type == 'ridge':
            return self._objective_ridge(X_train, y_train)
        elif self.model_type == 'elastic_net':
            return self._objective_elastic_net(X_train, y_train)
        elif self.model_type == 'lgbm':
            return self._objective_lgbm(X_train, y_train)
        elif self.model_type == 'xgboost':
            return self._objective_xgboost(X_train, y_train)
        elif self.model_type == 'catboost':
            return self._objective_catboost(X_train, y_train, categorical_features)
        elif self.model_type == 'random_forest':
            return self._objective_random_forest(X_train, y_train)
        elif self.model_type == 'prophet':
            if date_column is None or target_column is None:
                raise ValueError("date_column and target_column must be provided for Prophet")
            return self._objective_prophet(X_train, date_column, target_column, regressor_columns)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Parameters:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            float: MAPE score (lower is better)
        """
        # Filter out zeros in y_true to avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')  # Return infinity if all true values are zero
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return mape
    
    def _calculate_weighted_score(self, y_true, y_pred):
        """
        Calculate a weighted score combining RMSE, MAE, and MAPE.
        
        Parameters:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            float: Weighted score (lower is better)
        """
        # Calculate individual metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = self._calculate_mape(y_true, y_pred)
        
        # Normalize metrics to similar scale (optional, but helpful)
        # For this example, we'll use simple min-max scaling based on reasonable expected ranges
        # These normalization factors may need adjustment based on your specific dataset
        rmse_normalized = rmse / 100  # Assuming RMSE is typically in the 0-100 range
        mae_normalized = mae / 50     # Assuming MAE is typically in the 0-50 range
        mape_normalized = mape / 50   # Assuming MAPE is typically in the 0-50% range
        
        # Combine metrics using specified weights
        weighted_score = (
            self.metric_weights.get('rmse', 0.4) * rmse_normalized +
            self.metric_weights.get('mae', 0.4) * mae_normalized +
            self.metric_weights.get('mape', 0.2) * mape_normalized
        )
        
        return weighted_score
    
    def _calculate_score(self, y_true, y_pred):
        """
        Calculate score based on specified metric.
        
        Parameters:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            float: Score (higher is better for optimization)
        """
        if self.metric == 'rmse':
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mae':
            return -mean_absolute_error(y_true, y_pred)
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'mape':
            return -self._calculate_mape(y_true, y_pred)
        elif self.metric == 'weighted':
            return -self._calculate_weighted_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _objective_linear(self, X_train, y_train):
        """Objective function for Linear Regression using time-based validation"""
        def objective(trial):
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
            
            model = LinearRegressionModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_lasso(self, X_train, y_train):
        """Objective function for Lasso Regression using time-based validation"""
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 10000)
            }
            
            model = LassoRegressionModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_ridge(self, X_train, y_train):
        """Objective function for Ridge Regression using time-based validation"""
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 1e-6, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
            }
            
            model = RidgeRegressionModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_elastic_net(self, X_train, y_train):
        """Objective function for ElasticNet Regression using time-based validation"""
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 1000, 10000)
            }
            
            model = ElasticNetRegressionModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_lgbm(self, X_train, y_train):
        """Objective function for LightGBM using time-based validation"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            model = LGBMModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_xgboost(self, X_train, y_train):
        """Objective function for XGBoost using time-based validation"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            model = XGBoostModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_catboost(self, X_train, y_train, categorical_features=None):
        """Objective function for CatBoost using time-based validation"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'depth': trial.suggest_int('depth', 3, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0)
            }
            
            model = CatBoostModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y, cat_features=categorical_features)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_random_forest(self, X_train, y_train):
        """Objective function for Random Forest using time-based validation"""
        def objective(trial):
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'bootstrap': bootstrap
            }
            
            # Only include max_samples if bootstrap is True
            if bootstrap:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
                
            model = RandomForestModel(**params)
            
            # Create a single train/validation split
            train_X, train_y, val_X, val_y = self._create_time_split(X_train, y_train)
            
            # Train and evaluate on the validation set
            model.fit(train_X, train_y)
            y_pred = model.predict(val_X)
            score = self._calculate_score(val_y, y_pred)
            
            return score
        
        return objective
    
    def _objective_prophet(self, train_df, date_column, target_column, regressor_columns=None):
        """Objective function for Prophet using time-based validation"""
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.005, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.05, 20.0, log=True),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.7, 0.95),
                'n_changepoints': trial.suggest_int('n_changepoints', 20, 150),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            }
            
            try:
                # Create a single train/validation split
                # For Prophet, we need to create a synthetic target to use with _create_time_split
                synthetic_target = pd.Series(range(len(train_df)), index=train_df.index)
                train_X, _, val_X, _ = self._create_time_split(train_df, synthetic_target)
                
                # For Prophet, we need to have the target in the training data
                if target_column not in train_X.columns:
                    return float('-inf') if self.metric in ['r2'] else float('inf')
                
                model = ProphetModel(**params)
                model.fit(train_X, date_column, target_column, regressor_columns)
                y_pred = model.predict(val_X, date_column, regressor_columns)
                y_true = val_X[target_column].values
                
                score = self._calculate_score(y_true, y_pred)
                return score
            except Exception as e:
                self.logger.warning(f"Error in Prophet fitting: {e}")
                # Return a very poor score instead of failing
                return float('-inf') if self.metric in ['r2'] else float('inf')
        
        return objective
    
    def tune(self, X_train, y_train=None, categorical_features=None, 
             date_column=None, target_column=None, regressor_columns=None, experiment_name="default"):
        """
        Tune hyperparameters using Optuna.
        
        Parameters:
            X_train (DataFrame): Training features or full DataFrame for time series models
            y_train (array-like): Training targets (not needed for Prophet)
            categorical_features (list): List of categorical feature indices or names (for CatBoost)
            date_column (str): Date column name (for Prophet)
            target_column (str): Target column name (for Prophet)
            regressor_columns (list): List of regressor column names (for Prophet)
            experiment_name (str): Experiment name for saving best parameters
            
        Returns:
            dict: Best parameters
        """
        study_name = f"{self.model_type}_{self.metric}_study"
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        objective = self._get_objective_func(X_train, y_train, categorical_features, date_column, target_column, regressor_columns)
        
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        # Save best parameters to a JSON file
        self._save_best_params(experiment_name)
        
        return self.best_params
    
    def _save_best_params(self, experiment_name):
        """
        Save best parameters to a JSON file.
        
        Parameters:
            experiment_name (str): Experiment name for the file
        """
        if self.best_params is None:
            self.logger.warning("No best parameters to save.")
            return
            
        import json
        import os
        
        # Create best_params directory if it doesn't exist
        params_dir = "best_params"
        os.makedirs(params_dir, exist_ok=True)
        
        # Create filename
        filename = f"{self.model_type}_{experiment_name}.json"
        filepath = os.path.join(params_dir, filename)
        
        # Save parameters
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        self.logger.info(f"Best parameters saved to {filepath}")
    
    def load_best_params(self, experiment_name="default"):
        """
        Load best parameters from a JSON file.
        
        Parameters:
            experiment_name (str): Experiment name for the file
            
        Returns:
            dict: Best parameters
        """
        import json
        import os
        
        # Create filename
        filename = f"{self.model_type}_{experiment_name}.json"
        filepath = os.path.join("best_params", filename)
        
        try:
            with open(filepath, 'r') as f:
                self.best_params = json.load(f)
                self.logger.info(f"Loaded best parameters from {filepath}")
                return self.best_params
        except FileNotFoundError:
            self.logger.warning(f"File {filepath} not found. No parameters loaded.")
            return None
    
    def get_best_model(self):
        """
        Return the best model with optimized parameters.
        
        Returns:
            BaseRegressionModel: Model with best parameters
        """
        if self.best_params is None:
            raise ValueError("Tune hyperparameters first before getting the best model")
        
        if self.model_type == 'linear':
            return LinearRegressionModel(**self.best_params)
        elif self.model_type == 'lasso':
            return LassoRegressionModel(**self.best_params)
        elif self.model_type == 'ridge':
            return RidgeRegressionModel(**self.best_params)
        elif self.model_type == 'elastic_net':
            return ElasticNetRegressionModel(**self.best_params)
        elif self.model_type == 'lgbm':
            return LGBMModel(**self.best_params)
        elif self.model_type == 'xgboost':
            return XGBoostModel(**self.best_params)
        elif self.model_type == 'catboost':
            return CatBoostModel(**self.best_params)
        elif self.model_type == 'random_forest':
            return RandomForestModel(**self.best_params)
        elif self.model_type == 'prophet':
            return ProphetModel(**self.best_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def plot_optimization_history(self):
        """
        Plot the optimization history using Optuna visualization.
        
        Returns:
            plotly.Figure: Optimization history plot
        """
        if self.study is None:
            raise ValueError("Tune hyperparameters first before plotting")
        
        try:
            from optuna.visualization import plot_optimization_history
            return plot_optimization_history(self.study)
        except ImportError:
            self.logger.warning("Plotly is not installed. Cannot create visualization.")
            return None
    
    def plot_param_importances(self):
        """
        Plot the parameter importances using Optuna visualization.
        
        Returns:
            plotly.Figure: Parameter importances plot
        """
        if self.study is None:
            raise ValueError("Tune hyperparameters first before plotting")
        
        try:
            from optuna.visualization import plot_param_importances
            return plot_param_importances(self.study)
        except ImportError:
            self.logger.warning("Plotly is not installed. Cannot create visualization.")
            return None
    
    def get_model_with_default_params(self):
        """
        Return a model with default parameters.
        
        Returns:
            BaseRegressionModel: Model with default parameters
        """
        if self.model_type == 'linear':
            return LinearRegressionModel()
        elif self.model_type == 'lasso':
            return LassoRegressionModel()
        elif self.model_type == 'ridge':
            return RidgeRegressionModel()
        elif self.model_type == 'elastic_net':
            return ElasticNetRegressionModel()
        elif self.model_type == 'lgbm':
            return LGBMModel()
        elif self.model_type == 'xgboost':
            return XGBoostModel()
        elif self.model_type == 'catboost':
            return CatBoostModel()
        elif self.model_type == 'random_forest':
            return RandomForestModel()
        elif self.model_type == 'prophet':
            return ProphetModel()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


# Example usage
if __name__ == "__main__":
    # Example for LGBM
    # Load data
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # This is just a placeholder, replace with your actual data
    # df = pd.read_csv("your_data.csv")
    # X = df.drop(columns=["target"])
    # y = df["target"].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Initialize tuner
    # tuner = OptunaHyperparameterTuner(
    #     model_type='lgbm',
    #     metric='rmse',
    #     n_trials=50,
    #     validation_size=0.2,
    #     random_state=42
    # )
    
    # # Tune hyperparameters
    # best_params = tuner.tune(X_train, y_train)
    # print(f"Best parameters: {best_params}")
    
    # # Get best model
    # best_model = tuner.get_best_model()
    
    # # Train and evaluate
    # best_model.fit(X_train, y_train)
    # y_pred = best_model.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"RMSE with tuned hyperparameters: {rmse}")
    
    # # Plot optimization history
    # fig = tuner.plot_optimization_history()
    # if fig:
    #     fig.show()
    
    # # Plot parameter importances
    # fig = tuner.plot_param_importances()
    # if fig:
    #     fig.show()
    print("Import the OptunaHyperparameterTuner class to use it in your code.")
