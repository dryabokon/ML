import tools_IO
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
from TS import TS_AutoRegression, TS_ARIMA, TS_SARIMA, TS_BayesianRidge, TS_LSTM
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
    do_debug = True

    C1 = TS_AutoRegression.TS_AutoRegression(folder_out, do_debug=do_debug)
    C2 = TS_ARIMA.TS_ARIMA(folder_out, do_debug=do_debug)
    C3 = TS_SARIMA.TS_SARIMA(folder_out, do_debug=do_debug)
    C4 = TS_BayesianRidge.TS_BayesianRidge(folder_out, do_debug=do_debug)
    C5 = TS_LSTM.TS_LSTM(folder_out)
    ongoing_retrain = False

    for C in [C5]:
        TS = tools_TS.tools_TS(C, dark_mode=True, folder_out=folder_out)
        TS.E2E_train_test(df, idx_target=idx_target,ratio=0.85,ongoing_retrain=ongoing_retrain,do_debug=do_debug)

    return
# ----------------------------------------------------------------------------------------------------------------------
# df,idx_target  = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','),0
# df = df[['values','prev_values','sensor_day', 'time_on_day', 'day_of_week', 'id.1', 'categorical_id', 'hours_from_start', 'categorical_day_of_week']]
# ----------------------------------------------------------------------------------------------------------------------
# df,idx_target  = pd.read_csv(folder_in + 'air_quality.csv', delimiter=','),-1
# df = df.iloc[:1000,1:-2]
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out)
    #P.plot_target_feature(df, idx_target=idx_target)
    #P.plot_TS_separatly(df, idx_target=idx_target)
    df = tools_DF.hash_categoricals(df)
    ex_train_test(df, idx_target)










