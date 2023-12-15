import pandas as pd
import numpy as np

def create_forecast_frame(feature_data, forecast_horizon):
    start_date = pd.to_datetime(feature_data.index[-1]) + pd.DateOffset(days=1)
    date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
    forecast_df = pd.DataFrame(index=date_range, columns=["forecasted_value"])
    return forecast_df

def forecast(model, feature_data, forecast_temp, forecast_horizon, t_score, prediction_std):
    forecast_array = []
    columns_to_append = ['working_hours_lawyers', 'late_hours_lawyers', 'web_vis_proj']

    latest_data_row = feature_data.iloc[-1, 0:7].to_numpy().reshape(1, -1)

    for i in range(forecast_horizon):
        latest_data_row = np.append(latest_data_row, forecast_temp[columns_to_append].iloc[i])
        latest_data_row = latest_data_row.reshape(1, -1)
        predicted_result = model.predict(latest_data_row)

        new_data_row = np.concatenate((predicted_result, latest_data_row[0]))

        forecast_array.append(new_data_row)

        latest_data_row = new_data_row[0:7].reshape(1, -1)

    forecast_value = [array[0] for array in forecast_array]

    prediction_interval_lower = forecast_value - t_score * prediction_std
    prediction_interval_upper = forecast_value + t_score * prediction_std

    forecast_temp["forecasted_value"] = forecast_value
    forecast_temp["lower_value"] = prediction_interval_lower
    forecast_temp["upper_value"] = prediction_interval_upper

    forecast_final = forecast_temp[['forecasted_value', 'lower_value', 'upper_value']].copy()
    forecast_final = forecast_final.reset_index()
    forecast_final = forecast_final.rename(columns={'index':'date'})

    return forecast_final