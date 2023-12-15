import pandas as pd


def count_daily_consultations(df):
    count = df['created_at'].to_frame().reset_index(drop=True)
    count.set_index('created_at', inplace=True)
    df_daily_count = count.resample('D').size().reset_index()
    return df_daily_count

# Function to create lagged features for time series data
def create_lagged_features(data, lag):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'lag_{i}'] = data['count'].shift(i)
    data = lagged_data.fillna(lagged_data.mean(numeric_only=True))
    return data

def calculate_duration(df):
    df['duration'] = (df['end_datetime'] - df['start_datetime']).astype('timedelta64[m]') / 60
    return df

def extract_date(df, column):
    df['date'] = df[column].dt.date
    df['date'] = pd.to_datetime(df['date'])
    return df

def filter_working_hours(df):
    working_hours_mask = (df['start_datetime'].dt.hour >= 10) & (df['end_datetime'].dt.hour <= 18)
    return df[working_hours_mask]

def filter_late_hours(df):
    late_hours_mask = (df['end_datetime'].dt.hour > 18) & (df['end_datetime'].dt.hour <= 24)
    return df[late_hours_mask]

def get_daily_lawyer_count(df, time_column, count_column):
    return df.groupby(df[time_column].dt.date)[count_column].nunique()

def fill_date_range(daily_count):
    date_range = pd.date_range(start=daily_count.index.min(), end=daily_count.index.max(), freq='D')
    data = daily_count.reindex(date_range, fill_value=0).fillna(0)
    data = data.reset_index()
    return data

def create_df_lawyer_count(working_hours_lawyers, late_hours_lawyers):
    new_data = {'working_hours_lawyers': working_hours_lawyers, 'late_hours_lawyers': late_hours_lawyers}
    return pd.DataFrame(new_data)

def generate_date_range(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date)

def calculate_mean_count(df, count_column):
    return df[count_column].mean()

def create_new_data(date_range, mean_count):
    new_data = {'date': date_range, 'web_visitor': mean_count}
    return pd.DataFrame(new_data)

def concatenate_dataframes(df1, df2):
    return pd.concat([df1, df2]).sort_values('date')

def merge_dataframes(df1, df2, df3):
    # Merge the DataFrames on the 'date' column
    merged_df = pd.merge(df1, df2, on='date', how='outer')
    merged_df = pd.merge(merged_df, df3, on='date', how='outer')
    
    # Sort the DataFrame by 'date'
    merged_df = merged_df.sort_values('date')
    
    return merged_df