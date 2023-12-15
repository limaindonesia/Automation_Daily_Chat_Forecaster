import os
from etl.extract import get_data_consultation, get_data_availability, get_data_web_visitor
from etl.cleaning import *
from etl.transform import *
from utils import *

logger = get_logger(__name__)

def run(date_start: str= '2023-05-06', date_end: str= '2023-11-13'):

    # Set path
    relative_path = '../deep-flash-sa.json'
    file_path = os.path.abspath(relative_path)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path

    logger.info(f"Extracting data consultation from GCS.")
    data_consultation = get_data_consultation()
    logger.info(f"Successfully extracted data consultation from GCS.")
    
    logger.info(f"Extracting data data availability from GCS.")
    data_availability = get_data_availability()
    logger.info(f"Successfully extracted data data availability from GCS.")

    logger.info(f"Extracting data website visitor from local.")
    data_web = get_data_web_visitor() #need to improve this section from local to cloud
    logger.info(f"Extracting data website visitor from local.")

    logger.info(f"Transforming data consultation.")
    data_consultation_tr = transform_consultation(data_consultation)
    logger.info(f"Successfully transformed data consultation.")

    logger.info(f"Transforming data availability.")
    data_availability_tr = transform_availability(data_availability)
    logger.info(f"Successfully transformed data availibity.")

    logger.info(f"Transforming data website visitor.")
    data_web_tr = transform_web_visitor(data_web)
    logger.info(f"Successfully transformed data website visitor.")

    logger.info(f"Filtering data by date.")
    data_consultation_tr = filter_date_range(data_consultation_tr, date_start, date_end)
    data_availability_tr = filter_date_range(data_availability_tr, date_start, date_end)
    data_web_tr = filter_date_range(data_web_tr, date_start, date_end)
    logger.info(f"Successfully filtered data by date.")
    
    logger.info(f"Merging dataframes.")
    df_merged = merge_dataframes(data_consultation_tr, data_availability_tr, data_web_tr)
    logger.info(f"Successfully merged dataframes.")

    logger.info(f"Saving data to csv.")
    df_merged.to_csv('pipeline/feature_ml.csv', index=False)
    logger.info(f"Successfully saved data to csv.")

def transform_consultation(df):
    # cleaning
    df = remove_lawyer(df, [36, 38, 48, 120, 192, 195])
    # filter status 6000
    df = filter_status(df)
    # cast columns
    df = cast_column(df, ['created_at'])
    # count consultations
    df = count_daily_consultations(df)
    # rename column
    df = rename_columns(df, column_mapping={'created_at': 'date', 0: 'count'})
    # create lagged feature
    df = create_lagged_features(df, 7)
    return df
    
def transform_availability(df):
    # cleaning
    df = remove_lawyer(df, [36, 38, 48, 120, 192, 195])
    # filter column
    df = filter_columns(df, ['lawyer_id', 'start_datetime', 'end_datetime', 'created_at'])
    # cast columns
    df = cast_column(df, ['start_datetime', 'end_datetime', 'created_at'])
    # calculate duration
    df = calculate_duration(df)
    # extract date
    df = extract_date(df, 'created_at')
    # filter working hours
    df_working_hours = filter_working_hours(df)
    # filter late hours
    df_late_hours = filter_late_hours(df)
    # get daily lawyer count for working hours
    working_hours_lawyers = get_daily_lawyer_count(df_working_hours, 'start_datetime', 'lawyer_id')
    # get daily lawyer count for late hours
    late_hours_lawyers = get_daily_lawyer_count(df_late_hours, 'start_datetime', 'lawyer_id')
    # create daily count DataFrame
    df_new = create_df_lawyer_count(working_hours_lawyers, late_hours_lawyers)
    # fill date range
    df_new = fill_date_range(df_new)
    # rename
    df_new = rename_columns(df_new, column_mapping={'index': 'date'})
    return df_new
    
def transform_web_visitor(df):
    
    # rename column
    df = rename_columns(df, column_mapping={'event_date': 'date', 'f0_': 'web_visitor'})
    # convert date
    df = convert_date_to_datetime(df)
    # Generate date range
    date_range = generate_date_range('2023-05-06', '2023-06-25')
    # Calculate mean count
    mean_count = calculate_mean_count(df, 'web_visitor')
    # Create new data
    new_rows = create_new_data(date_range, mean_count)
    # Concatenate DataFrames
    df = concatenate_dataframes(df, new_rows)
    return df

run()