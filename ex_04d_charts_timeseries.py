import numpy
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,MultipleLocator
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=False)
# ----------------------------------------------------------------------------------------------------------------------
def get_raw_data():
    N = 100
    A = numpy.random.normal(loc=10.0, scale=10.0, size=(N,1))
    B = numpy.random.normal(loc= 8.0, scale=4.0 , size=(N,1))
    A[2:30,0]=numpy.nan
    B[17:49, 0] = numpy.nan

    idx = (N * numpy.random.random(30)).astype(int)
    A[idx] = numpy.nan

    return A,B
# ----------------------------------------------------------------------------------------------------------------------
def ex_multi_TS_range_num():
    A,B = get_raw_data()

    N = A.shape[0]
    idxA = numpy.linspace(10,150,N).astype('int32')
    idxB = numpy.linspace(30,160,N).astype('int32')

    df_A = pd.DataFrame(A, columns=['A'], index=idxA)
    df_B = pd.DataFrame(B, columns=['B'], index=idxB)
    df = pd.concat([df_A, df_B], axis=1)

    idxs_target = [0,1]

    df['idx'] = numpy.arange(0,df.shape[0])
    #df.to_csv(folder_out + 'multi_TS_num.csv',index=False)

    colors = tools_draw_numpy.get_colors(len(idxs_target),colormap='cool')
    for idx, clr in zip(idxs_target, colors):
        P.set_color(df.columns[idx], clr)

    out_format_x = ScalarFormatter(useOffset=False, useMathText=True)
    out_locator_x  = MultipleLocator(base=10)
    x_range = [0,df.shape[0]]

    P.TS_seaborn(df   , idxs_target=idxs_target, idx_time=-1,                                out_format_x=out_format_x,out_locator_x=out_locator_x,x_range=x_range,filename_out='multi_TS_num_pointplot_sns.png')
    P.TS_matplotlib(df, idxs_target=idxs_target, idx_time=-1, colors=colors[:,[2,1,0]]/255.0,out_format_x=out_format_x,out_locator_x=out_locator_x,x_range=x_range,filename_out='multi_TS_num_pointplot_mpl.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_multi_TS_range_date(df,idx_time,idxs_target):

    df.iloc[:,idx_time]=pd.to_datetime(df.iloc[:,idx_time])
    out_format_x = plt.matplotlib.dates.DateFormatter('%H:%M')

    start = datetime.datetime.combine(df.iloc[:,idx_time].min().date(),datetime.datetime.strptime('00:00:00', '%H:%M:%S').time())
    stop  = datetime.datetime.combine(df.iloc[:,idx_time].max().date(),datetime.datetime.strptime('23:00:00', '%H:%M:%S').time())

    colors = tools_draw_numpy.get_colors(len(idxs_target),colormap='warm')
    for idx, clr in zip(idxs_target,colors):
        P.set_color(df.columns[idx],clr)

    P.TS_seaborn   (df, idxs_target=idxs_target, idx_time=idx_time, mode='lineplot',               out_format_x=out_format_x,x_range=[start,stop],filename_out='multi_TS_date_pointplot_sns.png')
    P.TS_matplotlib(df, idxs_target=idxs_target, idx_time=idx_time, colors=colors[:,[2,1,0]]/255.0,out_format_x=out_format_x,x_range=[start,stop],filename_out='multi_TS_date_pointplot_mpl.png')


    return

# ----------------------------------------------------------------------------------------------------------------------
def ex_electricity():
    df = pd.read_csv(folder_in + 'dataset_electricity.csv', delimiter=',')
    idx_target = 1
    idx_time = 5

    df.iloc[:, idx_time] = pd.to_datetime(df.iloc[:, idx_time])

    df.iloc[numpy.random.choice(df.shape[0], int(df.shape[0] * 0.4)), idx_target] = numpy.nan
    df.iloc[:,idx_target]=100 + numpy.random.random(df.shape[0])*10
    df.iloc[-100:, idx_target] = numpy.nan

    out_locator_x = MultipleLocator(base=10)
    out_format_x = plt.matplotlib.dates.DateFormatter('%Y-%m-%d')

    # P.TS_seaborn(df, idx_target, None    , mode='pointplot',out_format_x=out_format_x,out_locator_x=out_locator_x,filename_out='seaborn_pointplot_idx.png')
    # P.TS_seaborn(df, idx_target, idx_time, mode='pointplot',out_format_x=out_format_x,out_locator_x=out_locator_x,filename_out='seaborn_pointplot_time.png')
    P.TS_seaborn(df, idx_target, idx_time, mode='lineplot' ,out_format_x=out_format_x,out_locator_x=out_locator_x,filename_out='seaborn_lineplot_time.png')


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # df = pd.read_csv('./data/ex_TS/air_quality.csv').iloc[:15]
    # idx_time = 0
    # idxs_target = [1, 2,-3]

    df = pd.read_csv(folder_in + 'st_stop.csv', delimiter=',')
    idx_time = 0
    idxs_target = [x for x in numpy.arange(1, df.shape[1])]

    #ex_multi_TS_range_num()
    ex_multi_TS_range_date(df,idx_time,idxs_target)
