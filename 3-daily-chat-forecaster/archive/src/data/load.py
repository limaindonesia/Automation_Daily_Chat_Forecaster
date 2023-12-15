import time
import pandas as pd
from google.cloud import storage
from io import StringIO


def read_data_from_gcs(bucket_name, folder, filename, delimiter=','):
    start_time = time.time()  # Start measuring time
    storage_client = storage.Client()
    blob = storage_client.get_bucket(bucket_name).blob(f'{folder}/{filename}')
    csv_data = blob.download_as_text()
    df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Read {filename} complete. Elapsed time: {elapsed_time:.2f} seconds")
    return df