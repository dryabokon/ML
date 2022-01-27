import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
from TS import TS_AutoRegression, TS_ARIMA, TS_SARIMA
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_TS
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
dark_mode = True
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=dark_mode)
# ----------------------------------------------------------------------------------------------------------------------
def ex_predict_ahead_once(df, idx_target,n_steps = 25):
    do_debug = True
    C1 = TS_AutoRegression.TS_AutoRegression(folder_out,do_debug=do_debug)
    C2 = TS_ARIMA.TS_ARIMA(folder_out,do_debug=do_debug)
    C3 = TS_SARIMA.TS_SARIMA(folder_out,do_debug=do_debug)

    for C in [C1,C2,C3]:
        TS = tools_TS.tools_TS(C, dark_mode=dark_mode, folder_out=folder_out)
        TS.predict_n_steps_ahead(df, idx_target,n_steps,do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_predict_ahead_animation(df, idx_target,n_steps = 25):

    n_start = df.shape[0] // 2

    tools_IO.remove_files(folder_out)

    C = TS_AutoRegression.TS_AutoRegression(folder_out, do_debug=False)
    TS = tools_TS.tools_TS(C, dark_mode=dark_mode, folder_out=folder_out)

    df_retro = pd.DataFrame({'GT': df.iloc[:n_start, idx_target],
                             'predict': numpy.full(n_start, numpy.nan),
                             'predict_ahead': numpy.full(n_start, numpy.nan),
                             'predict_ahead_min': numpy.full(n_start, numpy.nan),
                             'predict_ahead_max': numpy.full(n_start, numpy.nan),
                             })

    for limit in range(n_start, df.shape[0], 1):
        value_gt = float(df.iloc[limit, idx_target])
        df_step = TS.predict_n_steps_ahead(df.iloc[:limit], idx_target, n_steps)
        df_step['GT'] = numpy.full(n_steps, numpy.nan)
        df_step['predict'] =numpy.full(n_steps, numpy.nan)

        df_retro = df_retro.append(df_step, ignore_index=True)
        x_range = [max(0, df_retro.shape[0] - n_steps * 20), df_retro.shape[0]]
        P.TS_matplotlib(df_retro, [0, 2, 1], None, idxs_fill=[3, 4], x_range=x_range,filename_out='pred_ahead_%s_%04d.png' % (TS.classifier.name, df_retro.shape[0]))

        df_retro.drop(numpy.arange(df_retro.shape[0] - df_step.shape[0] + 1, df_retro.shape[0], 1), axis=0,inplace=True)
        df_retro['GT'].iloc[-1] = value_gt
        df_retro['predict'].iloc[-1] = df_retro['predict_ahead'].iloc[-1]
        df_retro['predict_ahead'] = numpy.nan
        print(df_retro.shape)

    return
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_predict_ahead_once(df, idx_target)
    #ex_predict_ahead_animation(df, idx_target)
