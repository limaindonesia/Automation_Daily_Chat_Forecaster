def get_date_range(df):
    """
    Get minimum and maximum date strings from a DataFrame column.
    
    Args:
        df (pandas DataFrame): DataFrame containing a 'date' column
        
    Returns:
        start (str): Minimum date string in YYYYMMDD format
        end (str): Maximum date string in YYYYMMDD format
    """
    
    first_date = df['date'].dt.strftime('%Y%m%d').min()
    curent_date = df['date'].dt.strftime('%Y%m%d').max()
    
    return first_date, curent_date

def save_df_to_csv(df, start, end, path):
    """
    Save a DataFrame to a CSV file with formatted filename.
    
    Args:
        df (pandas DataFrame): DataFrame to save
        start (str): Minimum date string in YYYYMMDD format
        end (str): Maximum date string in YYYYMMDD format
        path (str): Output path for CSV file
        
    Returns:
        None
    """
    
    filename = f"{path}/{start}-to-{end}.csv"
    df.to_csv(filename, index=False)
