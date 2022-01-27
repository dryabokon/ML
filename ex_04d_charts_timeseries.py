import numpy
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import seaborn
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
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
def ex_multi_TS_range_date(A,B):

    N = A.shape[0]
    delta = datetime.timedelta(minutes=0, seconds=1)
    start = datetime.datetime.strptime('2020-11-26 12:10:35', "%Y-%m-%d %H:%M:%S")
    idxA = [start + delta * t for t in numpy.linspace(10,150,N).astype('int')]
    idxB = [start + delta * t for t in numpy.linspace(30,160,N).astype('int')]

    df_A = pd.DataFrame(A,columns=['A'],index=idxA)
    df_B = pd.DataFrame(B,columns=['B'],index=idxB)
    df2  = pd.concat([df_A, df_B], axis=1)

    df2.to_csv(folder_out+'multi_TS_date.csv',sep='\t')
    P.TS_seaborn(df2, idxs_target=[0,1], idx_time=None, remove_xticks=False, major_step=10, filename_out='multi_TS_date_pointplot_sns.png')


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_multi_TS_range_num(A,B):

    N = A.shape[0]
    idxA = numpy.linspace(10,150,N).astype('int32')
    idxB = numpy.linspace(30,160,N).astype('int32')

    df_A = pd.DataFrame(A, columns=['A'], index=idxA)
    df_B = pd.DataFrame(B, columns=['B'], index=idxB)
    df2 = pd.concat([df_A, df_B], axis=1)

    df2.to_csv(folder_out + 'multi_TS_num.csv', sep='\t')

    P.TS_seaborn(df2, idxs_target=[0, 1], idx_time=None, remove_xticks=False, major_step=10, filename_out='multi_TS_num_pointplot_sns.png')


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_split(A):

    N = A.shape[0]
    delta = datetime.timedelta(minutes=0, seconds=1)
    start = datetime.datetime.strptime('2020-11-26 12:10:35', "%Y-%m-%d %H:%M:%S")
    idxA = [start + delta * t for t in numpy.linspace(10,150,N).astype('int')]
    df = pd.DataFrame(A,index=idxA)

    df1, df2 = train_test_split(df, test_size=0.6, shuffle=True)
    df1 = pd.concat([df, df1], axis=1,sort=True).iloc[:, df.shape[1]:]
    df2 = pd.concat([df, df2], axis=1,sort=True).iloc[:, df.shape[1]:]

    df_merge = pd.concat([df1, df2], axis=1)
    df_merge.columns=['1','2']
    df_merge['res'] = df_merge[['1', '2']].min(axis=1)
    df_merge = df_merge.drop(['1', '2'], axis=1)


    df.to_csv(folder_out + 'all.csv', sep='\t')
    df1.to_csv(folder_out + 'part1.csv', sep='\t')
    df2.to_csv(folder_out + 'part2.csv', sep='\t')
    df_merge.to_csv(folder_out + 'merged.csv', sep='\t')


    P.TS_seaborn(df, idxs_target=[0], idx_time=None, remove_xticks=False, major_step=10, filename_out='all.png')
    P.TS_seaborn(df1, idxs_target=[0], idx_time=None, remove_xticks=False, major_step=10, filename_out='part1.png')
    P.TS_seaborn(df2, idxs_target=[0], idx_time=None, remove_xticks=False, major_step=10, filename_out='part2.png')
    P.TS_seaborn(df_merge, idxs_target=[0], idx_time=None, remove_xticks=False, major_step=10, filename_out='merged.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_electricity():
    df,idx_target = pd.read_csv(folder_in + 'dataset_electricity.csv', delimiter=','),1
    df.iloc[numpy.random.choice(df.shape[0], int(df.shape[0] * 0.4)), idx_target] = numpy.nan
    idx_time = 5

    P.TS_seaborn(df, idx_target, None    , mode='pointplot',remove_xticks=False, major_step=100,filename_out='seaborn_pointplot_idx.png')
    P.TS_seaborn(df, idx_target, idx_time, mode='pointplot',remove_xticks=False, major_step=7,filename_out='seaborn_pointplot_time.png')
    P.TS_seaborn(df, idx_target, idx_time, mode='lineplot' ,remove_xticks=False, major_step=7, filename_out='seaborn_lineplot_time.png')


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    ex_electricity()

    # A,B = get_raw_data()
    # ex_multi_TS_range_num(A,B)
    # ex_multi_TS_range_date(A,B)

    #ex_split(A)