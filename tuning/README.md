# Hyperparameter Tuning

This module provides functionality for hyperparameter tuning using [Optuna](https://optuna.org/), a hyperparameter optimization framework.

## Overview

The `OptunaHyperparameterTuner` class is designed to optimize hyperparameters for various regression models. It uses Bayesian optimization to efficiently explore the hyperparameter space and find the best configuration for a given model.

## Supported Models

The tuner supports the following model types:

1. Linear regression models:
   - `linear`: Linear Regression
   - `lasso`: Lasso Regression
   - `ridge`: Ridge Regression
   - `elastic_net`: ElasticNet Regression

2. Tree-based models:
   - `lgbm`: LightGBM
   - `xgboost`: XGBoost
   - `catboost`: CatBoost
   - `random_forest`: Random Forest

3. Time series models:
   - `prophet`: Facebook Prophet

## Usage

Here's a basic example of how to use the hyperparameter tuner:

```python
from tuning.hyperparameter_tuning import OptunaHyperparameterTuner

# Initialize tuner
tuner = OptunaHyperparameterTuner(
    model_type='lgbm',
    metric='rmse',
    n_trials=100,
    cv=5,
    random_state=42
)

# Tune hyperparameters
best_params = tuner.tune(X_train, y_train, case_name="case1")

# Get the best model with optimized parameters
best_model = tuner.get_best_model()

# Train and predict
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
```

For Prophet models, which require different handling:

```python
# Initialize tuner for Prophet
tuner = OptunaHyperparameterTuner(
    model_type='prophet',
    metric='rmse',
    n_trials=50,
    cv=5,
    random_state=42
)

# Tune Prophet hyperparameters
best_params = tuner.tune(
    train_df,  # Pass the full DataFrame
    date_column='date',
    target_column='value',
    regressor_columns=['regressor1', 'regressor2'],  # Optional
    case_name="prophet_case1"
)

# Get the best model
best_model = tuner.get_best_model()

# Train and predict with Prophet
best_model.fit(train_df, 'date', 'value', regressor_columns)
predictions = best_model.predict(test_df, 'date', regressor_columns)
```

## Saving and Loading Parameters

The tuner automatically saves the best parameters to a JSON file in the `best_params` directory. The filename follows the pattern `modelname_casename.json`:

```python
# Tune with a specific case name
best_params = tuner.tune(X_train, y_train, case_name="production")
# Saved to: best_params/lgbm_production.json
```

You can also load previously saved parameters:

```python
# Load parameters for a specific case
tuner = OptunaHyperparameterTuner(model_type='lgbm')
loaded_params = tuner.load_best_params(case_name="production")

# Get a model with those parameters
model = tuner.get_best_model()
```

This makes it easy to:
1. Save optimized parameters for different datasets or scenarios
2. Reuse optimized parameters without having to run the optimization again
3. Compare performance between different parameter sets

## Visualization

The tuner provides methods to visualize the optimization process:

```python
# Plot optimization history
fig = tuner.plot_optimization_history()
fig.show()

# Plot parameter importances
fig = tuner.plot_param_importances()
fig.show()
```

## Examples

See `hyperparameter_example.py` for complete examples of tuning different types of models, including saving and loading parameters. 