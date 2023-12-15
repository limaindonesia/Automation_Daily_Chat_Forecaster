from sklearn.model_selection import TimeSeriesSplit

def split_timeseries_data(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = tscv.split(df)
    return splits