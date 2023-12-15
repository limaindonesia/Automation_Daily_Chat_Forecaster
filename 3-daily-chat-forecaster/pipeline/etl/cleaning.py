import pandas as pd


def remove_lawyer(df, lawyer_ids):
    filtered_df = df[~df['lawyer_id'].isin(lawyer_ids)]
    return filtered_df

def filter_status(df):
    filtered_df = df[df['status'] == 600]
    return filtered_df

def filter_columns(df, column):
    data = df[column]
    return data

def cast_column(df, columns):
    data = df.copy()
    for column in columns:
        data[column] = pd.to_datetime(data[column])
    return data

def convert_date_to_datetime(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def rename_columns(df, column_mapping):
    return df.rename(columns=column_mapping)

def filter_date_range(df, start_date, end_date):
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df[mask]