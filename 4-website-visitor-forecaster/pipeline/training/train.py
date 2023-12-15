from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import mlflow
import shutil
import os

def hyperparameter_tuning(X, y, cv_splits=5):
    # Time Series Split for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Hyperparameter tuning with cross-validation using MAE
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [1, 2, 3, 4],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'n_estimators': [100, 200, 300, 400, 500]
    }

    xgb = XGBRegressor(objective='reg:squarederror')

    # Define MAE as the scoring metric
    scoring_metric = make_scorer(mean_absolute_error, greater_is_better=False)

    # RandomizedSearchCV with TimeSeriesSplit
    random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, scoring=scoring_metric, cv=tscv)
    random_search.fit(X, y)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    print(f'Best Hyperparameters: {best_params}')

    return best_params

def train_and_evaluate(X, y, best_params, cv_splits=5):
    # Time Series Split for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Train XGBoost model with the best hyperparameters
    best_xgb_model = XGBRegressor(objective='reg:squarederror', **best_params)

    # Evaluate the model using cross-validation scores with MAE
    cv_scores = cross_val_score(best_xgb_model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
    mean_cv_score = np.mean(cv_scores)
    
    best_xgb_model.fit(X, y)
    # Display or log the results
    print(f'Cross-Validation Scores (MAE): {cv_scores}')
    print(f'Mean Cross-Validation Score (MAE): {mean_cv_score}')

    return best_xgb_model

def save_model(model, model_name):
    # Delete the existing model
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    mlflow.xgboost.save_model(model, model_name)