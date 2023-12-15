from inference.load import get_data_ads, load_model, load_data
from inference.process import convert_date_to_datetime, merge_data_by_index, set_date_to_index, save_result
from inference.forecast import create_forecast_frame, forecast
from utils import *

logger = get_logger(__name__)

def run():

    # google sheets 
    credential = "notebooks/ads-sheets-notebook-47e00ba6aad5.json"

    logger.info("Loading model...")
    loaded_model = load_model("best_xgb_model")
    logger.info("Model loaded!")

    logger.info("Loading data feature...")
    df = load_data("feature.csv")
    logger.info("Data loaded!")

    logger.info("Loading data ads projection")
    df_ads_proj = get_data_ads(sheet='ads_projection', credential=credential)
    logger.info("Data ads projection loaded!")

    logger.info("Converting date to datetime...")
    df_ads_proj = convert_date_to_datetime(df_ads_proj)
    logger.info("Date converted to datetime!")

    logger.info("Set date to index...")
    df_ads_proj = set_date_to_index(df_ads_proj)
    logger.info("Date set to index!")

    logger.info("Creating forecast frame...")
    forecast_df = create_forecast_frame(df, 21)
    logger.info("Forecast frame created!")

    logger.info("Merging data...")
    forecast_template = merge_data_by_index(forecast_df, df_ads_proj)
    logger.info("Data merged!")
    
    logger.info("Forecasting...")
    result = forecast(loaded_model, df, forecast_template, 21)
    logger.info("Forecasting finished!")

    logger.info("Saving result...")
    save_result(result)
    logger.info("Result saved!")

run()