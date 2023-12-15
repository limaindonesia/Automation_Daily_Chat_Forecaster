from statsmodels.tsa.api import ExponentialSmoothing
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet


# Model model
def model_exponential_smoothing(train_data, test_data):
    model = ExponentialSmoothing(train_data, seasonal="add", seasonal_periods=7)
    fit = model.fit()
    y_pred = fit.forecast(steps=len(test_data)) 
    return y_pred

def model_auto_arima(train_data, test_data):
    auto_model = auto_arima(train_data, seasonal=True, m=7, trace=True)
    n_forecast = len(test_data)
    y_pred, conf_int = auto_model.predict(n_forecast, return_conf_int=True)
    return y_pred

def model_linear_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
    
def model_lgb(X_train, y_train, X_test):
    model = lgb.LGBMRegressor(verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
    
def model_xgboost(X_train, y_train, X_test):
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def model_prophet(train_data_pr, test_data_pr):
    model = Prophet()
    model.fit(train_data_pr)
    y_pred = model.predict(test_data_pr)
    return y_pred['yhat']

