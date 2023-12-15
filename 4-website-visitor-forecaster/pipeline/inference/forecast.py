import pandas as pd
import numpy as np

def create_forecast_frame(feature_data, forecast_horizon):
    start_date = pd.to_datetime(feature_data.index[-1]) + pd.DateOffset(days=1)
    date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
    forecast_df = pd.DataFrame(index=date_range, columns=["forecasted_value"])
    return forecast_df

def forecast(model, feature_data, forecast_temp, forecast_horizon):
    forecast_array = []
    columns_to_append = ['cost_google', 'cost_meta']

    latest_data_row = feature_data.iloc[-1, 0:7].to_numpy().reshape(1, -1)

    for i in range(forecast_horizon):
        latest_data_row = np.append(latest_data_row, forecast_temp[columns_to_append].iloc[i])
        latest_data_row = latest_data_row.reshape(1, -1)
        predicted_result = model.predict(latest_data_row)

        new_data_row = np.concatenate((predicted_result, latest_data_row[0]))

        forecast_array.append(new_data_row)

        latest_data_row = new_data_row[0:7].reshape(1, -1)

    forecast_value = [array[0] for array in forecast_array]
    forecast_temp["forecasted_value"] = forecast_value
    
    return forecast_temp
