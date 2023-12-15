import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb

import mlflow
from training.split_timeseries import split_timeseries_data
from training.model import model_linear_regression, model_lgb, model_xgboost
from training.metrics import produce_metrics, metrics_results
from training.confidence import *
from utils import *

logger = get_logger(__name__)

def run():
    logger.info('Starting training pipeline...')
    logger.info('Reading feature data...')
    df = read_csv_and_set_index('pipeline/feature_ml.csv')
    logger.info('Successfully read feature data.')

    logger.info('Training Linear Regression...')
    score_lr = train_ml(df, 'Linear Regression')
    logger.info('Successfully trained Linear Regression.')

    logger.info('Training LightGBM...')
    score_lgbm = train_ml(df, 'LightGBM')
    logger.info('Successfully trained LightGBM.')

    logger.info('Training XGBoost...')
    score_xgb = train_ml(df, 'XGBoost')
    logger.info('Training XGBoost...')

    logger.info('Generating best model...')
    best_model_name = get_best_model([score_lr, score_lgbm, score_xgb], 'MAE')
    logger.info('Succesfully generated best model.')

    logger.info('Training best model...') 
    best_model = train_best_model(df, best_model_name)
    logger.info('Succesfully train best model.')

    logger.info('saving best model...')
    save_model(best_model, best_model_name)
    logger.info('Succesfully save best model.')


def read_csv_and_set_index(file_path, index_col='date'):
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)
    # Set the specified column as the index
    df.set_index(index_col, inplace=True)
    # Optionally, you can sort the DataFrame by the index if it's not already sorted
    df.sort_index(inplace=True)
    return df

def train_ml(df, model_name):
    models = {
        'Linear Regression': model_linear_regression,
        'LightGBM': model_lgb,
        'XGBoost': model_xgboost
    }
    
    scores = []
    splits = split_timeseries_data(df, n_splits=5)
    X = df.drop('count', axis=1)
    y = df['count']

    for _, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        y_pred = models[model_name](X_train, y_train, X_test)
        
        metrics = produce_metrics(y_test, y_pred)
        scores.append(metrics)
    
    average_score = metrics_results(scores)
    print(f"Print {model_name} score: {average_score}")

    # mlflow.set_tracking_uri("http://0.0.0.0:5001")
    # mlflow.set_experiment("exp-1")
    # with mlflow.start_run():
    #     mlflow.log_params({'model': model_name})
    #     mlflow.log_metrics({'MAE': average_score[0], 'RMSE': average_score[1], 'MAPE': average_score[2]})
        
    return average_score

def get_best_model(scores, metric):
    index = {'MAE': 0, 'RMSE': 1, 'MAPE': 2}
    
    if metric not in index:
        print("Invalid metric")
        return None
    
    metric_index = index[metric]
    min_value = min(score[metric_index] for score in scores)
    
    best_model_index = [i for i, score in enumerate(scores) if score[metric_index] == min_value][0]
    models = ['Linear Regression', 'LightGBM', 'XGBoost']  # Update with your actual model names
    best_model_name = models[best_model_index]

    print(f"Best model for {metric}: {best_model_name} with {metric} of {min_value}")
    return best_model_name

def train_best_model(df, best_model_name):
    
    X = df.drop('count', axis=1)
    y = df['count']
    
    if best_model_name == "Linear Regression":
        model = LinearRegression()
    elif best_model_name == "LightGBM":
        model = lgb.LGBMRegressor(verbose=-1)
    elif best_model_name == "XGBoost":
        model = xgb.XGBRegressor()
    else:
        raise ValueError("Invalid model name. Supported models: 'Linear Regression', 'LightGBM', 'XGBoost'")
    
    model.fit(X, y)
    y_pred = model.predict(X)

    t_score, prediction_std = interval_confidence(y, y_pred)
    save_interval_confidence(t_score, prediction_std, 'pipeline/t_&_std.csv')

    return model

def save_model(model, best_model_name):

    model_name = "best-model"
    # Delete the existing model
    if os.path.exists(model_name):
        import shutil
        shutil.rmtree(model_name)

    # Save the new model
    if best_model_name == "Linear Regression":
        mlflow.sklearn.save_model(model, model_name)
    elif best_model_name == "LightGBM":
        mlflow.lightgbm.save_model(model, model_name)
    elif best_model_name == "XGBoost":
        mlflow.xgboost.save_model(model, model_name)
    else:
        raise ValueError("Invalid model name. Supported models: 'Linear Regression', 'LightGBM', 'XGBoost'")
    
run()


