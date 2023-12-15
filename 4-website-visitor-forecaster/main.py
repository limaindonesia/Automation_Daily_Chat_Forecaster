# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression


# Create a Streamlit app with two pages
st.title("Website Visitor Forecast App")
df_google_proj = pd.read_csv('20231025-ads-google-proj.csv')
df_meta_proj = pd.read_csv('20231025-ads-meta-proj.csv')

df_fr = pd.read_csv('df_merged.csv')

st.subheader("Historical Data")
df_hist = df_fr.copy()
df_hist = df_hist[['date', 'cost_google', 'cost_meta']]
df_hist = st.data_editor(df_hist)

df_fr['date'] = pd.to_datetime(df_fr['date'])
df_fr.set_index('date', inplace=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Google Ads Spending Projection")
    df_google_proj = st.data_editor(df_google_proj)
    
with col2:
    st.subheader("Meta Ads Spending Projection")
    df_meta_proj = st.data_editor(df_meta_proj)


with open('lr_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# df_google_proj['cost'] = 100000

forecast_horizon = 73
start_date = df_fr.index.max() + pd.Timedelta(days=1)
date_range = pd.date_range(start_date, periods=forecast_horizon, freq='D')
forecast_df = pd.DataFrame(index=date_range, columns=["forecasted_value"])

forecast_array = []  # Renamed from 'forecast_value'
latest_data_row = df_fr.iloc[-1, 0:7].to_numpy().reshape(1, -1)

for i in range(forecast_horizon):
    latest_data_row_1 = np.append(latest_data_row, df_google_proj.iloc[i][1])
    latest_data_row_2 = np.append(latest_data_row_1, df_meta_proj.iloc[i][1])
    latest_data_row_3 = latest_data_row_2.reshape(1, -1)
    predicted_result = loaded_model.predict(latest_data_row_3)
    
    new_data_row = np.concatenate((predicted_result, latest_data_row_3[0]))
    print(new_data_row)
    forecast_array.append(new_data_row)
    
    latest_data_row = new_data_row[:-3].reshape(1, -1)
        
forecast_value = [array[0] for array in forecast_array]
forecast_df["forecasted_value"] = forecast_value


# Create a trace for actual values
trace_actual = go.Scatter(x=df_fr.index, y=df_fr['count'], mode='lines', name='Actual', line=dict(color='black'))

# Create a trace for forecasted values
trace_forecast = go.Scatter(x=forecast_df.index, y=forecast_df['forecasted_value'], mode='lines', name='Forecast', line=dict(color='green'))

# Create the layout for the plot
layout = go.Layout(
    title='Actual vs. Forecasted Values',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    legend=dict(x=0, y=1),
    showlegend=True
    # grid=True
)

# Create the figure
fig = go.Figure(data=[trace_actual, trace_forecast], layout=layout)

# Display the interactive Plotly figure using Plotly Express
st.plotly_chart(fig)