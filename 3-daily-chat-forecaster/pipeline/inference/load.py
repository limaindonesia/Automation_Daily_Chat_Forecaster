import pandas as pd
import mlflow

def load_model(model_name):
    return mlflow.sklearn.load_model(model_name)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.set_index('date')

def read_t_and_std_csv(file_path):
    t_and_std_data = pd.read_csv(file_path)
    t_score = t_and_std_data['t_score'][0]
    prediction_std = t_and_std_data['prediction_std'][0]
    return t_score, prediction_std

def load_web_visitor_forecast(file_path, start_date='2023-11-13'):
    web_vis = pd.read_csv(file_path)
    web_vis = web_vis.rename(columns={'Unnamed: 0':'date', 'forecasted_value':'web_vis_proj'})
    web_vis['date'] = pd.to_datetime(web_vis['date'])
    web_vis.set_index('date', inplace=True)
    web_vis = web_vis[web_vis.index > start_date]
    return web_vis