import pandas as pd

def merge_data_by_index(df1, df2):
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    return df

def add_lawyer_hours_features(forecast_df, working_hours=16, late_hours=5):
    """
    Add working hours and late hours features to the forecast DataFrame.
    """
    
    forecast_df['working_hours_lawyers'] = working_hours
    forecast_df['late_hours_lawyers'] = late_hours
    return forecast_df