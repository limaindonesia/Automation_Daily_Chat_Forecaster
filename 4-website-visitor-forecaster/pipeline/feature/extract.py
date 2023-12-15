import time
import pandas as pd
from google.cloud import storage
from io import StringIO
import gspread 
from oauth2client.service_account import ServiceAccountCredentials

#feature pipeline
def read_data_from_gcs(bucket_name, folder, filename, delimiter=','):
    start_time = time.time()  # Start measuring time
    storage_client = storage.Client()
    blob = storage_client.get_bucket(bucket_name).blob(f'{folder}/{filename}')
    csv_data = blob.download_as_text()
    df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Read {filename} complete. Elapsed time: {elapsed_time:.2f} seconds")
    return df

def get_data_webvisitor():
    data = read_data_from_gcs('perqara-dendrobium', 'website-visitor/processed', 'website-visitor-synthetic.csv')
    return data

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
    