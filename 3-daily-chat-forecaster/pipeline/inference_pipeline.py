from inference.load import load_model, load_data, read_t_and_std_csv, load_web_visitor_forecast
from inference.process import merge_data_by_index, add_lawyer_hours_features
from inference.forecast import create_forecast_frame, forecast
from utils import *

logger = get_logger(__name__)

def run():

    # Variable
    forecast_horizon = 48
    working_hours_lawyers = 16
    late_hourse_lawyers = 5

    logger.info('Loading model...')
    loaded_model = load_model("best-model") #path
    logger.info('Successfully loaded model.')

    logger.info('Loading feature data...')
    feature_data = load_data('pipeline/feature_ml.csv') #path
    logger.info('successfully loaded feature data.')

    logger.info('Loading t-score and prediction std...')
    t_score, prediction_std = read_t_and_std_csv('pipeline/t_&_std.csv') #path
    logger.info('Successfully loaded t-score and prediction std.')

    logger.info('Creating forecast frame...')
    forecast_df = create_forecast_frame(feature_data, forecast_horizon=forecast_horizon)
    logger.info('Successfully created forecast frame.')

    logger.info('Adding lawyer working hours and late hours features...')
    forecast_df = add_lawyer_hours_features(forecast_df, working_hours=working_hours_lawyers, late_hours=late_hourse_lawyers)
    logger.info('Successfully added lawyer working hours and late hours features.')

    logger.info('Loading web visitor forecast file...')
    web_visitor_forecast = load_web_visitor_forecast('pipeline/webvisitor_forecast_1109.csv', start_date='2023-11-13') #path
    logger.info('Successfully loaded web visitor forecast file.')

    logger.info('Merging data by index...')
    forecast_temp = merge_data_by_index(forecast_df, web_visitor_forecast)
    logger.info('Successfully merged data by index.')

    logger.info('Predicting...')
    forecast_result = forecast(loaded_model, feature_data, forecast_temp, forecast_horizon, t_score, prediction_std)
    logger.info('Successfully predicted.')
    
    logger.info('Saving forecast result...')
    forecast_result.to_csv('forecast_result.csv', index=True) #need to change this 
    logger.info('Successfully saved forecast result.')

run()