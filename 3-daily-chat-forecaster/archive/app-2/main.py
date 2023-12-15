import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import streamlit as st
import pickle
import joblib


#---------
#WEBSITE VISITOR SECTION
#add projection 
df_google_proj = pd.read_csv('20231025-ads-google-proj.csv') 
df_meta_proj = pd.read_csv('20231025-ads-meta-proj.csv')
df_google_proj = df_google_proj.rename(columns={'cost': 'cost_google'}, inplace=False)
df_meta_proj = df_meta_proj.rename(columns={'cost': 'cost_meta'}, inplace=False)
df_meta_proj= df_meta_proj[21:]
df_google_proj= df_google_proj[21:]
df_google_proj.set_index('date', inplace=True)
df_meta_proj.set_index('date', inplace=True)

# df_combined = pd.concat([df_google_proj, df_meta_proj], axis=1)
df_combined = pd.merge(df_meta_proj, df_google_proj, left_index=True, right_index=True)
st.sidebar.subheader('Ads Projected Cost')
df_combined = st.sidebar.data_editor(df_combined)

col_meta, col_google = st.columns(2)
with col_meta:
    meta_budget = st.sidebar.slider('Add Meta Budget', 0, 2000000, 100000)
with col_google:
    google_budget = st.sidebar.slider('Add Google Budget', 0, 2000000, 100000)

df_combined['cost_meta'] = df_combined['cost_meta'] + meta_budget
df_combined['cost_google'] = df_combined['cost_google'] + google_budget

feature_web = pd.read_csv('feature_web.csv')
feature_web['date'] = pd.to_datetime(feature_web['date'])
feature_web.set_index('date', inplace=True)

lr_model = joblib.load('linear_regression_model_web.pkl')

forecast_horizon = 52
start_date = feature_web.index.max() + pd.Timedelta(days=1)
date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
forecast_df_web = pd.DataFrame(index=date_range, columns=["forecasted_value"])

forecast_array = []  # Renamed from 'forecast_value'
latest_data_row = feature_web.iloc[-1, 0:7].to_numpy().reshape(1, -1)

for i in range(forecast_horizon):
    latest_data_row_1 = np.append(latest_data_row, df_combined['cost_google'].iloc[i])
    latest_data_row_2 = np.append(latest_data_row_1, df_combined['cost_meta'].iloc[i])
    latest_data_row_3 = latest_data_row_2.reshape(1, -1)
    predicted_result = lr_model.predict(latest_data_row_3)
    
    new_data_row = np.concatenate((predicted_result, latest_data_row_3[0]))
    print(new_data_row)
    forecast_array.append(new_data_row)
    
    latest_data_row = new_data_row[:-3].reshape(1, -1)
        
forecast_value = [array[0] for array in forecast_array]
forecast_df_web["forecasted_value"] = forecast_value
df_visitor_forecast = forecast_df_web[['forecasted_value']]

with st.sidebar:
    st.subheader('Website Visitor Projection')
    df_visitor_forecast = st.data_editor(forecast_df_web)

#---------
st.title("Daily Chat Consultation Forecast App")

#load model
loaded_model = joblib.load('linear_regression_model_with_bug.pkl')

#read data
df_final = pd.read_csv('df_final_with_bug.csv')
df_final.set_index('date', inplace=True)

#Lawyer working hours and late hours
st.sidebar.subheader('Daily Lawyer Online')
col1, col2 = st.columns(2)
with col1:
    working_hours = st.sidebar.slider('Working Hours Lawyers', 0, 50, 10)
with col2:
    late_hours = st.sidebar.slider('Late Hours Lawyers', 0, 50, 10)

# #web_visitor
# df_visitor_forecast = pd.read_csv('webvisitor_forecast_1109.csv')
# df_visitor_forecast.columns.values[0] = 'date'
# df_visitor_forecast = df_visitor_forecast[['forecasted_value']]

# Define the forecast horizon as a variable
forecast_horizon = 52
start_date = pd.to_datetime(df_final.index[-1]) + pd.DateOffset(days=1)
date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
forecast_df = pd.DataFrame(index=date_range, columns=["forecasted_value"])

end_time = pd.to_datetime(start_date) + pd.DateOffset(days=forecast_horizon - 1)

# Generate the time index
time_index = pd.date_range(start=start_date, end=end_time, freq='D')

# Create a sample DataFrame with this time index
df_add = pd.DataFrame(index=time_index)

# st.subheader("Projected Daily Lawyer Online until end of December 2023") 

df_add['working_hours_lawyers'] = working_hours
df_add['late_hours_lawyers'] = late_hours

cpa_google =  df_combined['cost_google'].values / df_visitor_forecast['forecasted_value'].values
cpa_meta = df_combined['cost_meta'].values / df_visitor_forecast['forecasted_value'].values
# st.write(cpa_me)

col3, col4 = st.sidebar.columns(2)
with col3:
    st.write('CPA Google')
    st.subheader(int(cpa_google.mean()))
with col4:
    st.write('CPA Meta')
    st.subheader(int(cpa_meta.mean()))
#cpa google

df_add['cpa_google'] = cpa_google
#cpa meta
df_add['cpa_meta'] = cpa_meta

# major and minor bug
df_add['cum_major_bug'] = 28
df_add['cum_minor_bug'] = 12

on = st.sidebar.toggle('Bug')
# minor_on = st.sidebar.toggle('Minor Bug')

if on:
    df_add['cum_major_bug'] = df_add['cum_major_bug'] + np.random.randint(0, 4, size=len(df_add))
    df_add['cum_minor_bug'] = df_add['cum_minor_bug'] + np.random.randint(0, 4, size=len(df_add))

inq = st.sidebar.number_input("Daily CR Inquiry:", value=190)
df_add['inq'] = inq

df_add['web_vis_proj'] = df_visitor_forecast

forecast_array = []  # Renamed from 'forecast_value'
latest_data_row = df_final.iloc[-1, 0:7].to_numpy().reshape(1, -1)

columns_to_append = ['web_vis_proj', 'working_hours_lawyers', 'late_hours_lawyers', 'cpa_google', 'cpa_meta', 'cum_major_bug', 'cum_minor_bug', 'inq']

# st.data_editor(df_add)

for i in range(forecast_horizon):
    # latest_data_row_1 = np.append(latest_data_row, df_visitor_forecast.iloc[i])
    # latest_data_row_2 = np.append(latest_data_row_1, df_add['working_hours'].iloc[i])
    # latest_data_row_3 = np.append(latest_data_row_2, df_add['late_hours'].iloc[i])
    # latest_data_row_4 = np.append(latest_data_row_3, df_add['cpa_google'].iloc[i])
    # latest_data_row_5 = np.append(latest_data_row_4, df_add['cpa_meta'].iloc[i])
    # latest_data_row_6 = np.append(latest_data_row_5, df_add['inq'].iloc[i])
    # latest_data_row_7 = np.append(latest_data_row_6, df_add['late_hours'].iloc[i])
    # latest_data_row_8 = np.append(latest_data_row_7, df_add['late_hours'].iloc[i])
    latest_data_row = np.append(latest_data_row, df_add[columns_to_append].iloc[i])

    latest_data_row = latest_data_row.reshape(1, -1)
    predicted_result = loaded_model.predict(latest_data_row)
    
    new_data_row = np.concatenate((predicted_result, latest_data_row[0]))
#     print(new_data_row)
    
    forecast_array.append(new_data_row)
    
    latest_data_row = new_data_row[0:7].reshape(1, -1)
    
forecast_value = [array[0] for array in forecast_array]
forecast_df["forecasted_value"] = forecast_value

#add confidence interval alpha = 0.2, the value got from notebook training
t_score = 1.3250575026979765
prediction_std = 9.360089857187747

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

