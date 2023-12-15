import pandas

def remove_lawyers(df, lawyer_ids):
    filtered_df = df[~df['lawyer_id'].isin(lawyer_ids)]
    return filtered_df

def get_completed_consultation(df, status):
    """
    Preprocess consultations DataFrame.
    
    Args:
        df (pandas DataFrame): Raw consultations data
        
    Returns:
        df (pandas DataFrame): Preprocessed consultations
    """

    df = df[df['status'] == 600]    
    return df

def resample_daily(df, date_col='created_at', count_col='count'):
    """
    Resample consultations to daily counts.
    
    Args:
        df (pandas DataFrame): Preprocessed consultations
        date_col (str): Date column name
        count_col (str): Count column name
        
    Returns:
        df (pandas DataFrame): Daily counts 
    """
    
    count = df[date_col].to_frame().reset_index(drop=True)
    count.set_index(date_col, inplace=True)
    df = count.resample('D').size().reset_index()
    df.rename(columns={date_col: 'date', 0: count_col}, inplace=True)
    
    return df