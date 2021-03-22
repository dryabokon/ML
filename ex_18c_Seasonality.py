import numpy
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
import statsmodels.api as sm
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def ex_decompose(ts):

    dates = numpy.array('2000-01-01', dtype=numpy.datetime64) + numpy.arange(ts.shape[0])
    df = pd.DataFrame({'data': ts.to_numpy()}, index=dates)

    plt.clf()
    plt.rcParams.update({'figure.figsize': (12, 7)})
    seasonal_decompose(df, model='multiplicative').plot().suptitle('Multiplicative Decomposition')
    plt.tight_layout()
    plt.savefig(folder_out + 'decompose2_mult.png')

    plt.clf()
    seasonal_decompose(df, model='additive').plot().suptitle('Additive Decomposition')
    plt.tight_layout()
    plt.savefig(folder_out + 'decompose2_add.png')

    plt.clf()
    autocorrelation_plot(ts)
    plt.savefig(folder_out + 'autocorrelation.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def do_lag_plot(X):
    for i in range(4):
        plt.clf()
        lag_plot(X, lag=i + 1, c='firebrick')
        plt.savefig(folder_out + 'lag_plot_%02d.png'%i)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_UnobservedComponents(df, idx_target):
    #https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_seasonal.html
    model = sm.tsa.UnobservedComponents(df.iloc[:, idx_target],
                                        level='fixed intercept',
                                        freq_seasonal=[{'period': 8,'harmonics': 2},
                                                       {'period': 48,'harmonics': 2}
                                                       ])
    res_f = model.fit(disp=False)
    plt.rcParams.update({'figure.figsize': (12, 7)})

    plt.clf()
    res_f.plot_components()
    plt.tight_layout()
    plt.savefig(folder_out + 'UnobservedComponents_stats.png')

    plt.clf()
    res_f.plot_diagnostics()
    plt.tight_layout()
    plt.savefig(folder_out + 'UnobservedComponents_diagnostics.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
#df,idx_target = pd.DataFrame({'data': 2+numpy.sin(numpy.arange(1000)/10)+numpy.linspace(0,5,1000)}, index=numpy.arange(1000)),0
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #P.TS_matplotlib(df, idx_target, None, filename_out='Target.png')
    #ex_decompose(df.iloc[:, idx_target])
    ex_UnobservedComponents(df, idx_target)
