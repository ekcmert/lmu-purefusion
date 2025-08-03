import numpy as np
import pandas as pd
from prophet import Prophet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


class BaseRegressionModel:
    """Base class for all regression models"""
    
    def __init__(self, model_name):
        """
        Initialize the base regression model.
        
        Parameters:
            model_name (str): Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        
        Parameters:
            X_train (array-like): Training features
            y_train (array-like): Training targets
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        
        Parameters:
            X_test (array-like): Test features
            
        Returns:
            numpy.array: Predicted values
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5, method='grid'):
        """
        Tune hyperparameters using cross-validation.
        
        Parameters:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            param_grid (dict): Dictionary of parameter values to try
            cv (int): Number of cross-validation folds
            method (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            
        Returns:
            dict: Best parameters
        """
        if self.model is None:
            raise ValueError("Model must be initialized before tuning hyperparameters")
        
        if method == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                verbose=1, n_jobs=-1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                self.model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                verbose=1, n_jobs=-1, n_iter=20
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        search.fit(X_train_scaled, y_train)
        
        self.model = search.best_estimator_
        return search.best_params_


class LinearRegressionModel(BaseRegressionModel):
    """Linear Regression model"""
    
    def __init__(self, normalize=True, fit_intercept=True):
        """
        Initialize the Linear Regression model.
        
        Parameters:
            normalize (bool): Whether to normalize the data (handled by StandardScaler)
            fit_intercept (bool): Whether to fit the intercept
        """
        super().__init__("linear_regression")
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
    
    def fit(self, X_train, y_train):
        """
        Fit the Linear Regression model.
        
        Parameters:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            
        Returns:
            self: The fitted model
        """
        # Use the same scaling approach as other models
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        
        Parameters:
            X_test (array-like): Test features
            
        Returns:
            numpy.array: Predicted values
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class LassoRegressionModel(BaseRegressionModel):
    """Lasso Regression model"""
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize the Lasso Regression model.
        
        Parameters:
            alpha (float): Regularization strength
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for the optimization
        """
        super().__init__("lasso_regression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class RidgeRegressionModel(BaseRegressionModel):
    """Ridge Regression model"""
    
    def __init__(self, alpha=1.0, solver='auto', tol=1e-4):
        """
        Initialize the Ridge Regression model.
        
        Parameters:
            alpha (float): Regularization strength
            solver (str): Solver for the optimization problem
            tol (float): Tolerance for the optimization
        """
        super().__init__("ridge_regression")
        self.alpha = alpha
        self.solver = solver
        self.tol = tol
        self.model = Ridge(
            alpha=alpha,
            solver=solver,
            tol=tol,
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class ElasticNetRegressionModel(BaseRegressionModel):
    """ElasticNet Regression model"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        Initialize the ElasticNet Regression model.
        
        Parameters:
            alpha (float): Regularization strength
            l1_ratio (float): Ratio of L1 penalty to the L1+L2 penalties
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for the optimization
        """
        super().__init__("elastic_net_regression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class LGBMModel(BaseRegressionModel):
    """LightGBM Regression model"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=40, max_depth=6, 
                 boosting_type='gbdt', min_child_samples=20, subsample=0.8, 
                 colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0):
        """
        Initialize the LightGBM Regression model.
        
        Parameters:
            n_estimators (int): Number of boosting iterations
            learning_rate (float): Learning rate
            num_leaves (int): Maximum number of leaves in one tree
            max_depth (int): Maximum tree depth
            boosting_type (str): Boosting type
            min_child_samples (int): Minimum number of data needed in a child (leaf)
            subsample (float): Subsample ratio of the training instances
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
            reg_alpha (float): L1 regularization term on weights
            reg_lambda (float): L2 regularization term on weights
        """
        super().__init__("lgbm")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.boosting_type = boosting_type
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.model = LGBMRegressor(
            random_state=42,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            boosting_type=boosting_type,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbose=-1
        )
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            numpy.array: Feature importances
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")
        return self.model.feature_importances_


class XGBoostModel(BaseRegressionModel):
    """XGBoost Regression model"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, 
                 colsample_bytree=0.8, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1):
        """
        Initialize the XGBoost Regression model.
        
        Parameters:
            n_estimators (int): Number of boosting iterations
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            subsample (float): Subsample ratio of the training instances
            colsample_bytree (float): Subsample ratio of columns for each tree
            min_child_weight (int): Minimum sum of instance weight needed in a child
            gamma (float): Minimum loss reduction required to make a further partition
            reg_alpha (float): L1 regularization term on weights
            reg_lambda (float): L2 regularization term on weights
        """
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.model = XGBRegressor(
            random_state=42,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbosity=0
        )
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            numpy.array: Feature importances
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")
        return self.model.feature_importances_


class CatBoostModel(BaseRegressionModel):
    """CatBoost Regression model"""
    
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
                 random_strength=1.0, bagging_temperature=1.0, border_count=128,
                 grow_policy='SymmetricTree'):
        """
        Initialize the CatBoost Regression model.
        
        Parameters:
            iterations (int): Number of boosting iterations
            learning_rate (float): Learning rate
            depth (int): Maximum tree depth
            l2_leaf_reg (float): L2 regularization coefficient
            random_strength (float): Amount of randomness to use for scoring splits
            bagging_temperature (float): Bagging temperature
            border_count (int): Number of splits for numerical features
            grow_policy (str): Controls how the tree is built - 'SymmetricTree', 'Depthwise', or 'Lossguide'
        """
        super().__init__("catboost")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.grow_policy = grow_policy
        
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            border_count=border_count,
            grow_policy=grow_policy,
            random_seed=42,
            verbose=0
        )
    
    def fit(self, X_train, y_train, cat_features=None):
        """
        Fit the CatBoost model.
        
        Parameters:
            X_train (DataFrame): Training features
            y_train (array-like): Training targets
            cat_features (list): List of categorical feature indices
        """
        # For CatBoost, we don't scale the features
        self.model.fit(X_train, y_train, cat_features=cat_features)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            numpy.array: Feature importances
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")
        return self.model.feature_importances_


class RandomForestModel(BaseRegressionModel):
    """Random Forest Regression model"""
    
    def __init__(self, n_estimators=100, max_depth=6, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, max_samples=None):
        """
        Initialize the Random Forest Regression model.
        
        Parameters:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees (was previously None/unlimited)
            min_samples_split (int, float): Minimum samples required to split a node
            min_samples_leaf (int, float): Minimum samples required at a leaf node
            max_features (str, int, float): Number of features to consider for best split
                                           ('sqrt' was previously 'auto' in older versions)
            bootstrap (bool): Whether to use bootstrap samples
            max_samples (float, int): Number of samples to draw from X to train each tree
                                    (Only if bootstrap=True)
        """
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        
        # Handle 'auto' vs 'sqrt' for different scikit-learn versions
        max_features_param = max_features
        if max_features == 'auto':
            # 'auto' is deprecated, use 'sqrt' instead in newer versions
            max_features_param = 'sqrt'
            
        # Only include max_samples if bootstrap is True
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features_param,
            'bootstrap': bootstrap,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if bootstrap and max_samples is not None:
            params['max_samples'] = max_samples
            
        self.model = RandomForestRegressor(**params)
    
    def fit(self, X_train, y_train):
        """
        Fit the Random Forest model.
        
        Parameters:
            X_train (array-like): Training features
            y_train (array-like): Training targets
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        """
        Make predictions using the trained Random Forest model.
        
        Parameters:
            X_test (array-like): Test features
            
        Returns:
            numpy.array: Predicted values
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            numpy.array: Feature importances
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")
        return self.model.feature_importances_


class ProphetModel(BaseRegressionModel):
    """Prophet Forecasting model"""
    
    def __init__(self, changepoint_prior_scale=0.1, seasonality_prior_scale=10.0, changepoint_range=0.9, 
                 n_changepoints=50, yearly_seasonality=False, weekly_seasonality=True, 
                 daily_seasonality=True, seasonality_mode='additive'):
        """
        Initialize the Prophet Forecasting model.
        
        Parameters:
            changepoint_prior_scale (float): Prior scale for trend changepoints
            seasonality_prior_scale (float): Prior scale for seasonality
            changepoint_range (float): Proportion of history for placing changepoints
            n_changepoints (int): Number of potential changepoints
            yearly_seasonality (bool): Whether to include yearly seasonality
            weekly_seasonality (bool): Whether to include weekly seasonality
            daily_seasonality (bool): Whether to include daily seasonality
            seasonality_mode (str): Mode for seasonality ('additive' or 'multiplicative')
        """
        super().__init__("prophet")
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_range = changepoint_range
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.model = None  # Will be initialized during fit
    
    def fit(self, train_df, date_column, target_column, regressor_columns=None):
        """
        Fit the Prophet model.
        
        Parameters:
            train_df (DataFrame): Training data
            date_column (str): Name of the date column
            target_column (str): Name of the target column
            regressor_columns (list): List of regressor column names
        """
        # Rename columns for Prophet
        df_train = train_df[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})
        
        # Initialize Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            changepoint_range=self.changepoint_range,
            n_changepoints=self.n_changepoints,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(name='2H', period=1/12, fourier_order=12)
        self.model.add_seasonality(name='4H', period=1/6, fourier_order=12)
        self.model.add_seasonality(name='8H', period=1/3, fourier_order=12)
        
        # Add regressors if specified
        if regressor_columns:
            df_train = self._prepare_regressor_data(train_df, df_train, regressor_columns)
            for col in regressor_columns:
                self.model.add_regressor(col)
        
        # Fit the model
        self.model.fit(df_train)
        return self
    
    def predict(self, test_df, date_column, regressor_columns=None):
        """
        Make predictions using the trained Prophet model.
        
        Parameters:
            test_df (DataFrame): Test data
            date_column (str): Name of the date column
            regressor_columns (list): List of regressor column names
            
        Returns:
            numpy.array: Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Rename date column
        df_test = test_df[[date_column]].rename(columns={date_column: 'ds'})
        
        # Add regressors if specified
        if regressor_columns:
            df_test = self._prepare_regressor_data(test_df, df_test, regressor_columns)
        
        # Make predictions
        forecast = self.model.predict(df_test)
        return forecast['yhat'].values
    
    def _prepare_regressor_data(self, source_df, target_df, regressor_columns):
        """
        Prepare regressor data for Prophet.
        
        Parameters:
            source_df (DataFrame): Source DataFrame with original data
            target_df (DataFrame): Target DataFrame for Prophet
            regressor_columns (list): List of regressor column names
            
        Returns:
            DataFrame: Prepared DataFrame with regressors
        """
        for col in regressor_columns:
            if col in source_df.columns:
                target_df[col] = source_df[col].copy()
                if source_df[col].dtype == 'object' or source_df[col].dtype.name == 'category':
                    # Convert categorical to numeric codes
                    target_df[col] = pd.factorize(target_df[col])[0]
        return target_df


class BaselineModel(BaseRegressionModel):
    """Baseline model using a shift-based approach for time series forecasting"""
    
    def __init__(self, shift=24):
        """
        Initialize the Baseline model.
        
        Parameters:
            shift (int): Number of indices to shift back for predictions
                         (e.g., 1 for previous hour, 24 for same hour yesterday)
        """
        super().__init__(f"baseline_shift{shift}")
        self.shift = shift
        self.values = None
    
    def fit(self, train_df, target_column):
        """
        Store the target values from the complete time series.
        
        Parameters:
            train_df (DataFrame): Complete time series (train+test)
            target_column (str): Name of the target column
        """
        self.values = train_df[target_column].values
        return self
    
    def predict(self, num_test_samples):
        """
        Make predictions by simply shifting the target values by the specified shift.
        
        This assumes the entire time series was provided during fit(), and we only
        need to predict the last num_test_samples values.
        
        Parameters:
            num_test_samples (int): Number of test samples to predict
            
        Returns:
            numpy.array: Predicted values for the test period
        """
        if self.values is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Total data length
        data_length = len(self.values)
        
        # We want to predict the last num_test_samples
        test_start_idx = data_length - num_test_samples
        
        # Create predictions array
        predictions = np.empty(num_test_samples)
        
        # For each test index, get the value from shift positions earlier
        for i in range(num_test_samples):
            # Index in the full series
            current_idx = test_start_idx + i
            
            # The lagged index to use for prediction
            lagged_idx = current_idx - self.shift
            
            # Use the lagged value if available
            if lagged_idx >= 0:
                predictions[i] = self.values[lagged_idx]
            else:
                # Not enough history, use the earliest available value 
                # with the same periodic pattern
                predictions[i] = self.values[0]
        
        return predictions 


class WeightedEnsembleModel(BaseRegressionModel):
    """Weighted Ensemble model combining multiple base models"""
    
    def __init__(self, base_models=None, weights=None, min_weight=0.05):
        """
        Initialize the Weighted Ensemble model.
        
        Parameters:
            base_models (dict): Dictionary of base models {name: model_instance}
            weights (dict): Optional dictionary of weights {name: weight} for each model
            min_weight (float): Minimum weight allowed for any model (0-1)
        """
        super().__init__("weighted_ensemble")
        self.base_models = base_models or {}
        self.weights = weights or {}
        self.min_weight = min_weight
        self.scaler = None  # Not needed for ensemble
        
        # Validate weights if provided
        if weights:
            self._validate_weights()
    
    def _validate_weights(self):
        """Validate the weights to ensure they sum to 1.0 and are all >= min_weight"""
        if not self.weights:
            return
            
        # Check if weights match base models
        if set(self.weights.keys()) != set(self.base_models.keys()):
            raise ValueError("Weights and base models must have the same keys")
            
        # Check that all weights are >= min_weight
        if any(w < self.min_weight for w in self.weights.values()):
            raise ValueError(f"All weights must be >= {self.min_weight}")
            
        # Check that weights sum to approximately 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def add_model(self, name, model):
        """
        Add a model to the ensemble.
        
        Parameters:
            name (str): Name of the model
            model (BaseRegressionModel): Model instance to add
        """
        self.base_models[name] = model
        # Set equal weights if no weights are specified yet
        if not self.weights:
            weight = 1.0 / len(self.base_models)
            self.weights = {model_name: weight for model_name in self.base_models}
        else:
            # Adjust weights to keep the same proportion for existing models
            # and assign min_weight to the new model
            old_sum = sum(self.weights.values())
            if name not in self.weights:
                # Assign min_weight to new model
                self.weights[name] = self.min_weight
                # Adjust existing weights proportionally
                scale_factor = (1.0 - self.min_weight) / old_sum
                for model_name in self.weights:
                    if model_name != name:
                        self.weights[model_name] *= scale_factor
    
    def set_weights(self, weights):
        """
        Set weights for the ensemble models.
        
        Parameters:
            weights (dict): Dictionary of weights {name: weight} for each model
        """
        self.weights = weights.copy()
        self._validate_weights()
    
    def fit(self, X_train, y_train):
        """
        Fit all base models in the ensemble.
        
        Parameters:
            X_train (array-like): Training features
            y_train (array-like): Training targets
        """
        if not self.base_models:
            raise ValueError("No base models in the ensemble")
            
        # Fit each base model
        for model_name, model in self.base_models.items():
            if not hasattr(model, 'fit'):
                raise ValueError(f"Model {model_name} does not have a fit method")
            model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X_test):
        """
        Make weighted predictions using all base models.
        
        Parameters:
            X_test (array-like): Test features
            
        Returns:
            numpy.array: Weighted predictions
        """
        if not self.base_models:
            raise ValueError("No base models in the ensemble")
            
        if not self.weights:
            raise ValueError("Weights not set")
            
        # Get predictions from each base model
        predictions = {}
        for model_name, model in self.base_models.items():
            if not hasattr(model, 'predict'):
                raise ValueError(f"Model {model_name} does not have a predict method")
            
            predictions[model_name] = model.predict(X_test)
        
        # Compute weighted average
        weighted_pred = np.zeros(len(X_test))
        for model_name, weight in self.weights.items():
            weighted_pred += weight * predictions[model_name]
        
        return weighted_pred
    
    def optimize_weights(self, X_val, y_val, method='grid', n_trials=50):
        """
        Optimize the weights of the ensemble using a validation set.
        
        Parameters:
            X_val (array-like): Validation features
            y_val (array-like): Validation targets
            method (str): Optimization method ('grid' or 'optuna')
            n_trials (int): Number of trials for Optuna optimization
            
        Returns:
            dict: Optimized weights
        """
        if not self.base_models:
            raise ValueError("No base models in the ensemble")
            
        # Get predictions from each base model on validation set
        val_predictions = {}
        for model_name, model in self.base_models.items():
            val_predictions[model_name] = model.predict(X_val)
        
        if method == 'grid':
            return self._optimize_weights_grid(val_predictions, y_val)
        elif method == 'optuna':
            return self._optimize_weights_optuna(val_predictions, y_val, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_weights_grid(self, predictions, y_true):
        """
        Optimize weights using a grid search approach.
        
        Parameters:
            predictions (dict): Dictionary of predictions from each model
            y_true (array-like): True validation targets
            
        Returns:
            dict: Optimized weights
        """
        # Import here to avoid circular imports
        try:
            from tuning.hyperparameter_tuning import OptunaHyperparameterTuner
            
            def _calculate_score(weights_list):
                # Convert weights list to dict
                weights_dict = {model_name: weight for model_name, weight 
                              in zip(self.base_models.keys(), weights_list)}
                
                # Compute weighted predictions
                weighted_pred = np.zeros(len(y_true))
                for model_name, weight in weights_dict.items():
                    weighted_pred += weight * predictions[model_name]
                
                # Calculate error (negative for maximization)
                rmse = -np.sqrt(np.mean((y_true - weighted_pred) ** 2))
                return rmse
            
            # Use grid search on linear space
            n_models = len(self.base_models)
            best_score = float('-inf')
            best_weights = None
            
            # Generate grid points
            grid_size = max(3, min(10, 2 ** n_models))  # Adjust grid size based on model count
            grid_points = np.linspace(self.min_weight, 1.0, grid_size)
            
            # For efficiency, for more than 3 models we'll use random sampling
            max_iterations = 10000
            iterations = 0
            
            if n_models <= 3:
                # Exhaustive grid search for 2-3 models
                for weights in itertools.product(grid_points, repeat=n_models):
                    iterations += 1
                    # Normalize weights to sum to 1
                    weights_sum = sum(weights)
                    norm_weights = [w / weights_sum for w in weights]
                    
                    # Evaluate
                    score = _calculate_score(norm_weights)
                    if score > best_score:
                        best_score = score
                        best_weights = norm_weights
            else:
                # Random sampling from Dirichlet distribution
                import numpy as np
                from scipy.stats import dirichlet
                
                # Sample weights from Dirichlet to get sum=1 property
                alpha = np.ones(n_models)  # Equal concentration params
                for _ in range(max_iterations):
                    iterations += 1
                    weights = dirichlet.rvs(alpha, size=1)[0]
                    
                    # Enforce minimum weight
                    if np.min(weights) < self.min_weight:
                        continue
                    
                    # Evaluate
                    score = _calculate_score(weights)
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                    
                    if iterations >= max_iterations:
                        break
            
            # Convert to dict
            optimized_weights = {model_name: weight for model_name, weight 
                               in zip(self.base_models.keys(), best_weights)}
            
            print(f"Optimized ensemble weights (grid search): {optimized_weights}")
            print(f"Best score (negative RMSE): {best_score}")
            
            # Update weights
            self.weights = optimized_weights
            return optimized_weights
                
        except ImportError:
            print("OptunaHyperparameterTuner not available, using equal weights")
            weight = 1.0 / len(self.base_models)
            self.weights = {model_name: weight for model_name in self.base_models}
            return self.weights
    
    def _optimize_weights_optuna(self, predictions, y_true, n_trials=50):
        """
        Optimize weights using Optuna.
        
        Parameters:
            predictions (dict): Dictionary of predictions from each model
            y_true (array-like): True validation targets
            n_trials (int): Number of trials for optimization
            
        Returns:
            dict: Optimized weights
        """
        try:
            import optuna
            
            def objective(trial):
                # Get weights using Optuna, ensuring they sum to 1.0
                # Use raw parameters and normalize them later to ensure sum=1
                model_names = list(self.base_models.keys())
                raw_weights = {}
                
                for model_name in model_names:
                    # Use log-scale for raw weights
                    raw_weights[model_name] = trial.suggest_float(
                        f"weight_{model_name}", 
                        0.1, 10.0, log=True
                    )
                
                # Normalize to sum to 1.0 while ensuring min_weight constraint
                weight_sum = sum(raw_weights.values())
                normalized_weights = {
                    model_name: raw_weight / weight_sum 
                    for model_name, raw_weight in raw_weights.items()
                }
                
                # Ensure min_weight constraint
                below_min = [name for name, w in normalized_weights.items() if w < self.min_weight]
                
                if below_min:
                    # Assign min_weight to models below threshold
                    min_total = len(below_min) * self.min_weight
                    remaining = 1.0 - min_total
                    
                    # Distribute remaining weight proportionally
                    above_min = [name for name in model_names if name not in below_min]
                    if not above_min:  # If all models would be below min, use equal weights
                        normalized_weights = {name: 1.0/len(model_names) for name in model_names}
                    else:
                        # Proportionally allocate remaining weight
                        above_min_sum = sum(raw_weights[name] for name in above_min)
                        for name in model_names:
                            if name in below_min:
                                normalized_weights[name] = self.min_weight
                            else:
                                normalized_weights[name] = (
                                    self.min_weight + 
                                    (raw_weights[name] / above_min_sum) * 
                                    (remaining - len(above_min) * self.min_weight)
                                )
                
                # Double-check and normalize again if needed
                w_sum = sum(normalized_weights.values())
                if not np.isclose(w_sum, 1.0, atol=1e-10):
                    for name in normalized_weights:
                        normalized_weights[name] /= w_sum
                
                # Compute weighted predictions
                weighted_pred = np.zeros(len(y_true))
                for model_name, weight in normalized_weights.items():
                    weighted_pred += weight * predictions[model_name]
                
                # Calculate error (negative for maximization)
                rmse = np.sqrt(np.mean((y_true - weighted_pred) ** 2))
                return rmse
            
            # Create and run optimization study
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            
            # Extract and normalize weights
            optimized_weights = {}
            raw_weights = {}
            
            for param, value in best_params.items():
                if param.startswith("weight_"):
                    model_name = param[len("weight_"):]
                    raw_weights[model_name] = value
            
            # Normalize to sum to 1.0
            weight_sum = sum(raw_weights.values())
            optimized_weights = {
                model_name: raw_weight / weight_sum 
                for model_name, raw_weight in raw_weights.items()
            }
            
            # Handle min_weight constraint
            below_min = [name for name, w in optimized_weights.items() if w < self.min_weight]
            if below_min:
                # Recalculate weights to respect min_weight
                min_total = len(below_min) * self.min_weight
                remaining = 1.0 - min_total
                above_min = [name for name in self.base_models.keys() if name not in below_min]
                
                if above_min:
                    above_min_sum = sum(raw_weights[name] for name in above_min)
                    for name in optimized_weights:
                        if name in below_min:
                            optimized_weights[name] = self.min_weight
                        else:
                            proportion = raw_weights[name] / above_min_sum
                            optimized_weights[name] = (
                                self.min_weight + 
                                proportion * (remaining - len(above_min) * self.min_weight)
                            )
                else:
                    # If all models would be below min, use equal weights
                    equal_weight = 1.0 / len(self.base_models)
                    optimized_weights = {name: equal_weight for name in self.base_models}
            
            # Ensure exact normalization
            w_sum = sum(optimized_weights.values())
            if not np.isclose(w_sum, 1.0, atol=1e-10):
                for name in optimized_weights:
                    optimized_weights[name] /= w_sum
            
            print(f"Optimized ensemble weights (Optuna): {optimized_weights}")
            print(f"Best RMSE: {study.best_value}")
            
            # Update weights
            self.weights = optimized_weights
            return optimized_weights
            
        except ImportError:
            print("Optuna not available, using equal weights")
            weight = 1.0 / len(self.base_models)
            self.weights = {model_name: weight for model_name in self.base_models}
            return self.weights

# Import at module level
import itertools 