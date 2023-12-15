import pandas as pd

def load_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def prepare_data(df):
    # Prepare features and target
    X = df.drop(['count'], axis=1)
    y = df['count']
    return X, y