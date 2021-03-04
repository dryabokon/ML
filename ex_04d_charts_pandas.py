import numpy
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def ex_single_TS():
    df = pd.read_csv(folder_in + 'dataset_electricity.csv', delimiter=',')

    idx = numpy.random.choice(df.shape[0],int(df.shape[0]*0.4))
    idx_target, idx_feature = 1, None
    df.iloc[idx, idx_target] = numpy.nan

    P.TS_matplotlib(df, idx_target, idx_feature,filename_out='electricity_matplotlib.png')
    P.TS_seaborn(df,idx_target   ,idx_feature,filename_out='electricity_seaborn_pointplot.png')
    P.TS_seaborn(df, idx_target, idx_feature,mode='lineplot',filename_out='electricity_seaborn_lineplot.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_multi_TS():

    N= 100
    A = numpy.random.normal(loc=10.0, scale=10.0, size=(N,1))
    B = numpy.random.normal(loc= 8.0, scale=4.0, size=(N,1))
    df = pd.DataFrame(data=numpy.hstack((A, B)), columns=['A','B'])
    P.TSs_seaborn(df, idxs_target=[0,1], idx_feature=None, filename_out='multi_TS.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_single_TS()
    #ex_multi_TS()
