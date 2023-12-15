import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st

st.title("Daily Chat Consultation Forecast App")

#load model
loaded_model = joblib.load('linear_regression_model.pkl')

#read data
df_final = pd.read_csv('df_final.csv')
df_final.set_index('date', inplace=True)

#web_visitor
df_visitor_forecast = pd.read_csv('webvisitor_forecast_2.csv')
df_visitor_forecast.columns.values[0] = 'date'
with st.sidebar:
    st.subheader('Website Visitor Forecast')
    df_visitor_forecast = st.data_editor(df_visitor_forecast)
df_visitor_forecast = df_visitor_forecast[['forecasted_value']]


# Define the forecast horizon as a variable
forecast_horizon = 73
start_date = pd.to_datetime(df_final.index[-1]) + pd.DateOffset(days=1)
date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
forecast_df = pd.DataFrame(index=date_range, columns=["forecasted_value"])

end_time = pd.to_datetime(start_date) + pd.DateOffset(days=forecast_horizon - 1)

# Generate the time index
time_index = pd.date_range(start=start_date, end=end_time, freq='D')

# Create a sample DataFrame with this time index
data = {'onboarded_shift_1_week': [7] * len(time_index)}  # Creating a column with constant values for example
df_add = pd.DataFrame(data, index=time_index)

#lawyer online
df_add['lawyer_online_count'] = df_add.index.weekday < 5 

# st.subheader("Projected Daily Lawyer Online until end of December 2023") 
col1, col2, col3 = st.columns(3)

with col1:
    df_add['onboarded_shift_1_week'] = st.slider('Weekly Onboarded Lawyer', 0, 30, 7)
with col2:
    weekday_value = st.slider('Daily Lawyer Online Weekday', 0, 100, 10)
with col3:
    weekend_value = st.slider('Daily Lawyer Online Weekend', 0, 100, 3)

df_add['lawyer_online_count'] = df_add['lawyer_online_count'].apply(lambda x: weekday_value if x else weekend_value)

forecast_array = []  # Renamed from 'forecast_value'
latest_data_row = df_final.iloc[-1, 0:7].to_numpy().reshape(1, -1)

for i in range(forecast_horizon):
    latest_data_row_1 = np.append(latest_data_row, df_visitor_forecast.iloc[i])
    latest_data_row_2 = np.append(latest_data_row_1, df_add.iloc[i][0])
    latest_data_row_3 = np.append(latest_data_row_2, df_add.iloc[i][1])
    latest_data_row_4 = latest_data_row_3.reshape(1, -1)
    predicted_result = loaded_model.predict(latest_data_row_4)
    
    new_data_row = np.concatenate((predicted_result, latest_data_row_4[0]))
#     print(new_data_row)
    
    forecast_array.append(new_data_row)
    
    latest_data_row = new_data_row[:-4].reshape(1, -1)
    
forecast_value = [array[0] for array in forecast_array]
forecast_df["forecasted_value"] = forecast_value

#add confidence interval alpha = 0.2, the value got from notebook training
t_score = 1.3261412063716906
prediction_std = 10.7334146469441

prediction_interval_lower = forecast_df["forecasted_value"] - t_score * prediction_std
prediction_interval_upper = forecast_df["forecasted_value"] + t_score * prediction_std

forecast_df['lower'] = forecast_df["forecasted_value"] - t_score * prediction_std
forecast_df['upper'] = forecast_df["forecasted_value"] + t_score * prediction_std

# Create a trace for actual values
trace_actual = go.Scatter(x=df_final.index, y=df_final['count'], mode='lines', name='Actual', line=dict(color='black'))

# Create a trace for forecasted values
trace_forecast = go.Scatter(x=forecast_df.index, y=forecast_df['forecasted_value'], mode='lines', name='Forecast', line=dict(color='green'))

# Calculate upper and lower confidence intervals
upper_confidence = forecast_df['upper']  # Replace with your actual upper confidence data
lower_confidence = forecast_df['lower']  # Replace with your actual lower confidence data

# Create traces for upper and lower confidence intervals
trace_upper_confidence = go.Scatter(x=forecast_df.index, y=upper_confidence, fill=None, mode='lines', line=dict(color='rgba(189, 253, 170, 0.3)'), name='Upper Confidence')
trace_lower_confidence = go.Scatter(x=forecast_df.index, y=lower_confidence, fill='tonexty', mode='lines', line=dict(color='rgba(189, 253, 170, 0.3)'), name='Lower Confidence')

# Create the layout for the plot
layout = go.Layout(
    title='Actual vs. Forecasted Values with Confidence Intervals',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1),
    showlegend=True
)

# Create the figure
fig = go.Figure(data=[trace_actual, trace_forecast, trace_upper_confidence, trace_lower_confidence], layout=layout)

# Display the interactive Plotly figure using Plotly Express
st.plotly_chart(fig)

previous_value = df_final['count'].sum()
forecasted_value = int(previous_value + forecast_df["forecasted_value"].sum())
upper_value = int(previous_value + upper_confidence.sum())
lower_value = int(previous_value + lower_confidence.sum())

st.write("Total Projected Consultations by the end of December 2023") 

col1, col2, col3 = st.columns(3)
col1.metric(label="Forecasted Value", value=forecasted_value)
col2.metric(label="Optimistic", value=upper_value)
col3.metric(label="Pessimistic", value=lower_value)


# forecast_df_new = forecast_df.reset_index()
# forecast_df_new['month'] = forecast_df_new['index'].dt.month
# november = forecast_df_new[forecast_df_new['month'] == 11]
# desember = forecast_df_new[forecast_df_new['month'] == 12]


# st.write('November')
# st.write((november['forecasted_value'].sum()))

# st.write('Desember')
# st.write((desember['forecasted_value'].sum()))

