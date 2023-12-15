from sklearn.impute import SimpleImputer

def extract_date(df):
    date = df['date']
    return date

def data_process_stats(df):
    time_series = df['count']
    return time_series

def fill_na(df):
    df = df.fillna(df.mean(numeric_only=int))
    return df

# Function to create lagged features for time series data
def create_lagged_features(data, lag):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'Lag_{i}'] = data['count'].shift(i)

    lagged_data = fill_na(lagged_data)
    return lagged_data

    
def data_process_prophet(df):
    df = df.rename(columns={'date': 'ds', 'count': 'y'}, inplace=False)
    return df


def data_process_ml(df, train_index, test_index):
    
    df = df.set_index('date', inplace=False)

    lag = 7  # Number of lagged values, adjust as needed
    # Apply the function to create lagged features
    lagged_df = create_lagged_features(df, lag)
    
    X = lagged_df.drop(['count'], axis=1)
    y = lagged_df['count']
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    
    return X_train, y_train, X_test, y_test