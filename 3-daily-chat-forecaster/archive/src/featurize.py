import os
import pandas as pd
import datetime
import yaml

from features.build_features import extract_date, data_process_stats, create_lagged_features, data_process_prophet
from utils.logging import get_logger

#import configs
config_path = '/Users/fadilrisdian/perqara-projects/3-daily-chat-forecaster/config/params.yaml'
config = yaml.safe_load(open(config_path))
log_level = config['base']['log_level']
logger = get_logger("FEATURIZE", log_level)

# 1. load data
logger.info('Load dataset')
current_directory = os.getcwd()
first_date = config['data']['first_date']
current_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d") 
file = f'{first_date}-to-{current_date}.csv'
df = pd.read_csv(f'{current_directory}/3-daily-chat-forecaster/data/raw/{file}')

#2. Extract Date
logger.info('Extract Date')
date = extract_date(df)

#Process data for arima and es model
logger.info('Extract Feature for arima and es model')
data_stats = data_process_stats(df)

#Process data for machine learning model
logger.info('Extract Feature for machine learning model')
data_ml = create_lagged_features(df, 7)

#Process data for prophet model
logger.info('Extract Feature for prophet model')
data_pr = data_process_prophet(df)

#Save to csv
logger.info('save feature to csv')
date.to_csv(current_directory + '/3-daily-chat-forecaster/data/processed/date.csv', index=False)
data_stats.to_csv(current_directory + '/3-daily-chat-forecaster/data/processed/data_stats.csv', index=False)
data_ml.to_csv(current_directory + '/3-daily-chat-forecaster/data/processed/data_ml.csv', index=False)
data_pr.to_csv(current_directory + '/3-daily-chat-forecaster/data/processed/data_pr.csv', index=False)

