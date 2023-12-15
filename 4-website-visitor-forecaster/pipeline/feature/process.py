import pandas as pd

def filter_date(df, start, end):
    df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    return df

def convert_date_to_datetime(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

def filter_column(df):
    df = df[['date','cost_google', 'cost_meta']].copy()
    return df 
 
def convert_columns_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype(int)
    return df

def create_lagged_features(data, lag):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'lag_{i}'] = data['count'].shift(i)
    data = lagged_data.fillna(lagged_data.mean(numeric_only=True))
    return data
        
def merge_dataframes(df1, df2):
    merged_df = pd.merge(df1, df2, on='date', how='outer')
    merged_df = merged_df.sort_values('date')
    return merged_df

def save_result(df):
    df = df.reset_index()
    df.to_csv("result.csv")