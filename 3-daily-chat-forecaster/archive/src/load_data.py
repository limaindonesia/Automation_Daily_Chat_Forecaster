import os
import pandas as pd
import yaml

from data.load import read_data_from_gcs
from data.process import remove_lawyers, get_completed_consultation, resample_daily
from utils.util import get_date_range, save_df_to_csv
from utils.logging import get_logger

# import configs
config_path = '/Users/fadilrisdian/perqara-projects/3-daily-chat-forecaster/config/params.yaml'
config = yaml.safe_load(open(config_path))

log_level = config['base']['log_level']
bucket_name = config['gcs_data_load']['bucket_name']
path = config['gcs_data_load']['path']  
filename = config['gcs_data_load']['filename']
testing_user = config['process']['testing_user']

logger = get_logger("DATA_LOAD", log_level)

# 1. load data from google cloud storage
logger.info('Load dataset')
current_directory = os.getcwd()
relative_path = current_directory + "/deep-flash-sa.json"
file_path = os.path.abspath(relative_path)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path
df = read_data_from_gcs(bucket_name, path, filename, delimiter='|')

# 2. remove testing user
logger.info('Remove Lawyers')
df = remove_lawyers(df, testing_user)

# 3. get completed consultations
logger.info('Get completed consultations')
df = get_completed_consultation(df, 600)

# 4. change date format
logger.info('Change Date Format')
df['created_at'] = pd.to_datetime(df['created_at'])

# 5. resample consultations to daily counts
logger.info('Resample to daily')
df = resample_daily(df, date_col='created_at', count_col='count')

# 6. save data to CSV
logger.info('Save feature to csv')
start, end = get_date_range(df)
save_df_to_csv(df, start, end, '3-daily-chat-forecaster/data/raw')


