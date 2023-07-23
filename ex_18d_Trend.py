import numpy
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_video
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=False)
# -----------------------------------------------------------------------# ----------------------------------------------------------------------------------------------------------------------
#df,idx_target = pd.DataFrame({'data': numpy.random.random(10000)}, index=numpy.arange(10000)),0
#df,idx_target = pd.DataFrame({'data': numpy.sin(numpy.arange(1000)/10)}, index=numpy.arange(1000)),0
#df,idx_target = pd.DataFrame({'data': numpy.sin(numpy.arange(1000)/10)+numpy.linspace(0,5,1000)}, index=numpy.arange(1000)),0
#df,idx_target = sm.datasets.sunspots.load_pandas().data,1
#df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'daily-total-female-births.txt', delimiter='\t'), 1

df, idx_target = pd.read_csv(folder_in + 'Monthly_passengers.txt', delimiter='\t'), 1
#df, idx_target = pd.read_csv(folder_in + 'yapu.txt', delimiter='\t'), 0
# ----------------------------------------------------------------------------------------------------------------------
def detrend_v1(X):
    detrended = signal.detrend(X)
    return detrended
# ----------------------------------------------------------------------------------------------------------------------
def detrend_v2(X):
    detrended = X - seasonal_decompose(X, model='multiplicative',period=30).trend
    return detrended
# ----------------------------------------------------------------------------------------------------------------------
def detrend_v3(X):
    df = pd.DataFrame({'X':X})

    # df['diff'] = df['X'] - df['X'].shift(1)
    # res = df['diff'].values

    df['log'] = numpy.log(df['X'])
    df['log_diff'] = df['log'] - df['log'].shift(1)
    res = df['log_diff'].values


    A = 0.5*numpy.std(df['X'].dropna().values) / numpy.std(df['log_diff'].dropna().values)
    res*=A

    return res
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X = df.iloc[:, idx_target].to_numpy()
    df2 = pd.DataFrame({'original': X,
                        'detrend_v1 scipy': detrend_v1(X),
                        'detrend_v2 seasonal_decompose': detrend_v2(X),
                        'detrend_v3 diff log': detrend_v3(X)})


    P.TS_seaborn(df2, idxs_target=[0,1,2],idx_time=None,remove_legend=False,lw=3,x_range=None,y_range=None, figsize=(8,6), filename_out='detrend.png')