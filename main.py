import yfinance as yf
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.TransformationModel import TransformationModel

def spearman_test(df, period_label, predict_label):
    coef, _ = spearmanr(df[period_label], df[predict_label])

    return coef < 0

def adfuller_test(data):
    result = adfuller(data)

    return result[1] < 0.05

def get_seasonal_decomposition(df, predict_label, model_type, period):
    seasonal_decomposition = seasonal_decompose(df[predict_label], model=model_type, period=period)

    return seasonal_decomposition

def get_decomposition_forecast(decomposition):
    decomposition_trend = get_decomposition_trend(decomposition)
    decomposition_seasonality = decomposition.seasonal.dropna()
    decomposition_wast = decomposition.resid.dropna()
    decomposition_forecast = decomposition_trend + decomposition_seasonality
    decomposition_forecast.dropna(inplace=True)

    return decomposition_forecast

def get_decomposition_trend(decomposition):
    decomposition_trend = decomposition.trend.dropna()

    return decomposition_trend

def get_real_data(df, predict_label, decomposition):
    decomposition_trend = get_decomposition_trend(decomposition)

    real_data = df[predict_label][decomposition_trend.index]

    return real_data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_rmse(data, forecast):
    rmse = np.sqrt(mean_squared_error(data, forecast))

    return rmse

def get_mae(data, forecast):
    mae = mean_absolute_error(real_data, additive_forecast)

    return mae

def get_best_transformation(data, transformations):
    best_rmse = float('inf')
    best_mae = float('inf')
    best_mape = float('inf')
    best_transformation = None

    for transformation in transformations:
        data_aligned, transformation_aligned = data.align(transformation.get_transformation(), join='inner')

        rmse = get_rmse(data_aligned, transformation_aligned)
        mae = get_mae(data_aligned, transformation_aligned)
        mape = mean_absolute_percentage_error(data_aligned, transformation_aligned)

        if rmse < best_rmse:
            best_rmse = rmse
            best_transformation = transformation

    return best_transformation

yf_start = "2014-08-27"
yf_end = "2024-08-27"
yf_interval = "1mo"

brent_df = yf.download('BZ=F', start=yf_start, end=yf_end, interval=yf_interval).reset_index()

brent_df["DateIndex"] = [i for i, _ in enumerate(brent_df["Date"])]

df_predict_label = "Close"

spearman_test_result = spearman_test(brent_df, "DateIndex", df_predict_label)

adfuller_test_result = adfuller_test(brent_df[df_predict_label])

seasonal_decomposition_period = 12

additive_decomposition = get_seasonal_decomposition(brent_df, df_predict_label, "additive", seasonal_decomposition_period)
multiplicative_decomposition = get_seasonal_decomposition(brent_df, df_predict_label, "multiplicative", seasonal_decomposition_period)

additive_forecast = get_decomposition_forecast(additive_decomposition)

multiplicative_forecast = get_decomposition_forecast(multiplicative_decomposition)

real_data = get_real_data(brent_df, df_predict_label, additive_decomposition)

rmse_additive = get_rmse(real_data, additive_forecast)
mae_additive = get_mae(real_data, additive_forecast)
mape_aditivo = mean_absolute_percentage_error(real_data, additive_forecast)

rmse_multiplicative = get_rmse(real_data, multiplicative_forecast)
mae_multiplicative = get_mae(real_data, multiplicative_forecast)
mape_multiplicative = mean_absolute_percentage_error(real_data, multiplicative_forecast)

transformations = []

brent_df_diff = brent_df[df_predict_label].diff()

transformations.append(TransformationModel("diferenciada", brent_df_diff))

brent_df_diff.dropna(inplace=True)

brent_df_diff_adfuller_test_result = adfuller_test(brent_df_diff)

brent_df_log = np.log(brent_df[df_predict_label])

transformations.append(TransformationModel("log", brent_df_log))

brent_df_log_adfuller_test_result = adfuller_test(brent_df_log)

brent_df_diff_seasonal = brent_df[df_predict_label].diff()

transformations.append(TransformationModel("diferenciada sazonal", brent_df_diff_seasonal))

brent_df_diff_seasonal.dropna(inplace=True)
brent_df_diff_seasonal_adfuller_test_result = adfuller_test(brent_df_diff_seasonal)

brent_df_diff_second_order = brent_df[df_predict_label].diff().dropna()
brent_df_diff_second_order_adfuller_test_result = adfuller_test(brent_df_diff_second_order)

brent_df_log_diff = pd.DataFrame(brent_df_log).diff()

transformations.append(TransformationModel("log diferenciada", brent_df_log_diff))

brent_df_log_diff.dropna(inplace=True)

brent_df_log_diff_adfuller_test_result = adfuller_test(brent_df_log_diff)

transformation = get_best_transformation(brent_df[df_predict_label], transformations)

print(transformation.get_name())