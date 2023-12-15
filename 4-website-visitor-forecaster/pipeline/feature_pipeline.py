import os
from feature.extract import get_data_webvisitor, get_data_ads
from feature.process import filter_date, convert_date_to_datetime, create_lagged_features, filter_column, convert_columns_to_int, merge_dataframes, save_result
from utils import *

logger = get_logger(__name__)

def run():

    # google cloud sheets
    relative_path = '../deep-flash-sa.json'
    file_path = os.path.abspath(relative_path)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path
    # google sheets credential
    credential = "notebooks/ads-sheets-notebook-47e00ba6aad5.json"

    # read data 
    logger.info("Extracting data website visitor from GCS.")
    df_website_visitor = get_data_webvisitor()
    logger.info("Successfully extracted data website visitor from GCS")

    logger.info("Extracting data ads spending from google sheets.")
    df_website_ads = get_data_ads(sheet='ads', credential=credential)
    logger.info("Extracting data ads spending from google sheets.")

    # transform website
    logger.info("Transforming data website visitor.")
    df_web_visitor = filter_date(df_website_visitor, '2023-07-01', '2023-12-10')
    logger.info("Successfully transformed data website visitor.")

    logger.info("Converting date to datetime.")
    df_web_visitor = convert_date_to_datetime(df_web_visitor)
    logger.info("Successfully converted date to datetime.")

    logger.info("Creating lagged features.")
    df_web_visitor = create_lagged_features(df_web_visitor, 7)
    logger.info("Successfully created lagged features.")

    # transform ads
    logger.info("Transforming data ads.")
    df_ads = filter_column(df_website_ads)
    logger.info("Successfully transformed data ads.")

    logger.info("Converting date to datetime.")
    df_ads = convert_date_to_datetime(df_ads)
    logger.info("Successfully converted date to datetime.")

    logger.info("Converting column to int.")
    df_ads = convert_columns_to_int(df_ads, ['cost_google', 'cost_meta'])
    logger.info("Successfully converting column to int.")

    # merge data
    logger.info("Merging data.")
    merged_df = merge_dataframes(df_web_visitor, df_ads)
    logger.info("Successfully merged data.")

    logger.info("Saving result.")
    save_result(merged_df)
    logger.info("Successfully saved result.")

run()