import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
from TS import TS_AutoRegression, TS_ARIMA, TS_SARIMA
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
def ex_train_test(df, idx_target):
    do_debug = False

    C1 = TS_AutoRegression.TS_AutoRegression(folder_out, do_debug=do_debug)
    # C2 = TS_ARIMA.TS_ARIMA(folder_out, do_debug=do_debug)
    # C3 = TS_SARIMA.TS_SARIMA(folder_out, do_debug=do_debug)

    for C in [C1]:
        TS = tools_TS.tools_TS(C, dark_mode=True, folder_out=folder_out)
        TS.E2E_train_test(df, idx_target=idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_train_test(df, idx_target)

