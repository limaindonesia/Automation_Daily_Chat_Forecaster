import time
import pandas as pd
from io import StringIO
from google.cloud import storage


def read_data_from_gcs(bucket_name, folder, filename, delimiter=','):
    start_time = time.time()  # Start measuring time
    storage_client = storage.Client()
    blob = storage_client.get_bucket(bucket_name).blob(f'{folder}/{filename}')
    csv_data = blob.download_as_text()
    df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Read {filename} complete. Elapsed time: {elapsed_time:.2f} seconds")
    return df

def get_data_consultation():
    data = read_data_from_gcs('perqara-dendrobium', 'raw/postgres/csv/consultations', 'consultations.csv', delimiter='|')
    return data

def get_data_availability():
    data = read_data_from_gcs('perqara-dendrobium', 'raw/postgres/csv/availability_instants', 'availability_instants.csv', delimiter=',')
    return data
    
def get_data_web_visitor():
    data = pd.read_csv('pipeline/website-visitor_20231114.csv')
    return data