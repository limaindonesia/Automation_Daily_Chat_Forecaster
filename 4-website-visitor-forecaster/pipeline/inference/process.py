import pandas as pd

def convert_date_to_datetime(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

def merge_data_by_index(df1, df2):
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    return df

def set_date_to_index(df): 
    df = df.set_index('date')
    return df

def save_result(df):
    df = df.reset_index()
    df.to_csv("result.csv")