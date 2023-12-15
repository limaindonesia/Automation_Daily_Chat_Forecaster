import pandas as pd
import numpy as np

def interval_confidence(y_test, y_pred, confidence=0.8):
    alpha = 1 - confidence
    n = len(y_test)
    t_score = np.abs(np.percentile(np.random.standard_t(df=n-2, size=10000), 100 * (1 - alpha / 2)))

    residuals = y_test - y_pred
    mse = np.mean(residuals ** 2)
    prediction_std = np.sqrt(mse)
    
    return t_score, prediction_std

def save_interval_confidence(t_score, prediction_std, path):
    pd.DataFrame({'t_score': [t_score], 'prediction_std': [prediction_std]}).to_csv(path, index=False)