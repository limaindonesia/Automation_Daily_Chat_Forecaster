import pandas as pd
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import mlflow

def load_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def get_data_ads(sheet, credential):
    spreadsheet_key = "1KZJmFezoZlc9drr-y5grwWE3u9UMqMSzAafzq89Ky7w"
    scope = "https://spreadsheets.google.com/feeds"
    start_time = time.time()  # Start measuring time
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credential, scope)
    worksheet = gspread.authorize(credentials).open_by_key(spreadsheet_key).worksheet(sheet)
    data = worksheet.get_all_values()
    headers = data.pop(0)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Read {sheet} sheet complete. Elapsed time: {elapsed_time:.2f} seconds")
    return pd.DataFrame(data, columns=headers)

def load_model(model_name):
    return mlflow.xgboost.load_model(model_name)