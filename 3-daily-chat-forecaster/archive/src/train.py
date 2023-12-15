import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import yaml

from models.model import model_exponential_smoothing, model_auto_arima, model_linear_regression, model_lgb, model_xgboost, model_prophet
from evaluate.metrics import produce_metrics, metrics_results
from utils.logging import get_logger

#import configs
config_path = '/Users/fadilrisdian/perqara-projects/3-daily-chat-forecaster/config/params.yaml'
config = yaml.safe_load(open(config_path))
log_level = config['base']['log_level']
logger = get_logger("TRAIN", log_level)

path = '/Users/fadilrisdian/perqara-projects/3-daily-chat-forecaster/data/processed/'
n_splits = config['train']['n_splits']
tscv = TimeSeriesSplit(n_splits=n_splits)

models = ['es', 'arima', 'lr', 'lgb', 'xgb', 'pr']
metric_results = []

for model in models:
    if model == 'es' or model == 'arima':
        df = pd.read_csv(path+ 'data_stats.csv')
        
    if model == 'lr' or model == 'lgb' or model == 'xgb':
        df = pd.read_csv(path+ 'data_ml.csv')

    if model == 'pr':
        df = pd.read_csv(path+ 'data_pr.csv')

    # Initialize a list to store the mean squared errors for each fold
    scores = [] 

    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"Fold {i}:")
        print(train_index)
        print(test_index)

        if model == 'es' or model == 'arima':
            train_data, y_test = df.iloc[train_index], df.iloc[test_index]
            if model == 'es':
                y_pred = model_exponential_smoothing(train_data, y_test)
            if model == 'arima':
                y_pred = model_auto_arima(train_data, y_test)
                
        if model == 'lr' or model == 'lgb' or model == 'xgb':
            df = pd.read_csv(path+ 'data_ml.csv')
            df = df.drop(['date'], axis=1)
            X = df.drop(['count'], axis=1)
            y = df['count']

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            if model == 'lr':
                y_pred = model_linear_regression(X_train, y_train, X_test)
            if model == 'lgb':
                y_pred = model_lgb(X_train, y_train, X_test)
            if model == 'xgb':
                y_pred = model_xgboost(X_train, y_train, X_test)

        if model == 'pr':
            train_data_pr, y_test = df.iloc[train_index], df.iloc[test_index]
            y_pred = model_prophet(train_data_pr, y_test)
            y_test = y_test['y']

        # Calculate the MAE, RMSE, MAPE
        metrics = produce_metrics(y_test, y_pred)

        # Append metrics score
        scores.append(metrics)
        
        #Print RMSE for each fold
        print(f"Fold RMSE {model}: {metrics}")
        
        avg_scores = metrics_results(scores)
        
    metric_results.append(avg_scores)

# Create a Pandas DataFrame
df_model = pd.DataFrame({'Model': ['Exponential Smoothing', 'ARIMA', 'Linear Regression', 'LightGBM', 'XGBoost','Prophet']})
df_res = pd.DataFrame(metric_results)
df_res.columns = ['MAE', 'RMSE', 'MAPE']
df_final_scores = pd.concat([df_model, df_res], axis=1)

df_date = pd.read_csv(path+'date.csv')
df_date['date'] = pd.to_datetime(df_date['date'])
start = df_date['date'].dt.strftime('%Y%m%d').min()
end = df_date['date'].dt.strftime('%Y%m%d').max()

df_final_scores.to_csv(f'/Users/fadilrisdian/perqara-projects/3-daily-chat-forecaster/reports/ml_scores-{start}-{end}.csv', index=False)