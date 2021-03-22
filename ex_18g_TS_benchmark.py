from matplotlib import pyplot as plt
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.api import ARIMA
from matplotlib import pyplot
import statsmodels.api as sm
import numpy
import pandas as pd
import itertools
# ----------------------------------------------------------------------------------------------------------------------
from TS import TS_Naive,TS_AutoRegression,TS_LinearRegression,TS_BayesianRidge, TS_ARIMA
# ----------------------------------------------------------------------------------------------------------------------
import tools_TS
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
def ex_sarima():
    y = df.iloc[:,idx_target].to_numpy()

    # p = d = q = range(0, 2)
    # for param in list(itertools.product(p, d, q)):
    #     for param_seasonal in [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]:
    #         model = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
    #         results = model.fit()
    #         print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


    model = sm.tsa.statespace.SARIMAX(y,order=(1, 1, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
    fitted = model.fit(disp=False)
    fitted.plot_diagnostics(figsize=(15, 12))
    plt.savefig(folder_out+'diagnostics.png')

    prediction = fitted.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
    pred_ci = prediction.conf_int()
    #
    # ax = y['1990':].plot(label='observed')
    # pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
    #
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.2)
    #
    # ax.set_xlabel('Date')
    # ax.set_ylabel('CO2 Levels')
    # plt.legend()
    #
    #
    # plt.savefig(folder_out+'predict.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_train_test(df, idx_target)
    ex_predictions_ahead(df, idx_target)

