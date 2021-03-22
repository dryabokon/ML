import numpy
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def test_adfuller(ts):

    #constant
    dftest = adfuller(ts,regression="c")
    result = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():result['Critical Value (%s)'%key] = value

    p_value = result['p-value']
    a_significance_level = 0.05

    #H0 = non-stationary
    if p_value <= a_significance_level:
        print("p value <= significance_level")
        print('%1.2f    < %1.2f' % (p_value, a_significance_level))
        print('Stationary OK')#Reject H0: s
    else:
        print("significance_level < p value")
        print('%1.2f               < %1.2f' % (a_significance_level, p_value))
        print('Non-stationary')#Accept H0:
    print()
    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
def test_kpss(ts):

    kpsstest = kpss(ts, regression='c')
    result = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():result['Critical Value (%s)'%key] = value

    p_value = result['p-value']
    a_significance_level = 0.05
    #H0 = stationary
    if p_value <= a_significance_level:
        print("p value <= significance_level")
        print('%1.2f    < %1.2f' % (p_value, a_significance_level))
        print('Non-stationary')#Reject H0
    else:
        print("significance_level < p value")
        print('%1.2f               < %1.2f' % (a_significance_level, p_value))
        print('Stationary OK')#Accept H0
    print()
    print()


    return
# ----------------------------------------------------------------------------------------------------------------------
#df,idx_target = pd.DataFrame({'data': numpy.random.random(10000)}, index=numpy.arange(10000)),0
#df,idx_target = pd.DataFrame({'data': numpy.sin(numpy.arange(1000)/10)}, index=numpy.arange(1000)),0
#df,idx_target = pd.DataFrame({'data': numpy.sin(numpy.arange(1000)/10)+numpy.linspace(0,5,1000)}, index=numpy.arange(1000)),0
#df,idx_target = sm.datasets.sunspots.load_pandas().data,1
#df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'Monthly_passengers.txt', delimiter='\t'), 1
df, idx_target = pd.read_csv(folder_in + 'daily-total-female-births.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
# df, idx_target = pd.read_csv(folder_in + 'YX_sine.txt', delimiter='\t'), 0
# df.iloc[:, idx_target]+=numpy.linspace(0,2,df.shape[0])
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    P.TS_matplotlib(df, idx_target, None, filename_out='Target.png')
    test_adfuller(df.iloc[:,idx_target])
    test_kpss(df.iloc[:, idx_target])

    smt.graphics.plot_acf(df.iloc[:, idx_target])
    plt.savefig(folder_out+'acf.png')
