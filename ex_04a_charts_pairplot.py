import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_pairplot_df(df,idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    target = columns[idx_target]
    idx =numpy.delete(numpy.arange(0,len(columns)),idx_target)
    vars = columns[idx]

    pal = numpy.array(['tab10', 'husl','Set2','Paired','hls'])[0]

    seaborn.pairplot(data=df, hue=target,vars=vars)
    plt.savefig(folder_out+'pairplot.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_pairplots_df(df0,idx_target=0):

    columns = df0.columns.to_numpy()
    target = columns[idx_target]
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    pal = numpy.array(['tab10', 'husl', 'Set2', 'Paired', 'hls'])[0]

    for i in range(len(idx)-1):
        for j in range(i+1,len(idx)):
            c1, c2 = columns[idx[i]], columns[idx[j]]
            df = df0[[target, c1, c2]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)

            plt.clf()
            seaborn.jointplot(data=df, x=c1, y=c2, hue=target,palette=pal)
            plt.savefig(folder_out + 'pairplot_%02d_%02d_%s_%s.png'%(i,j,c1,c2))
            plt.close()

    for i in range(len(idx)):
        c = columns[idx[i]]
        df = df0[[target, c]]
        df = df.dropna()
        df = tools_DF.hash_categoricals(df)
        seaborn.histplot(data=df, x=c, hue=target, palette=pal,element='poly')
        plt.savefig(folder_out + 'pairplot_%02d_%02d_%s.png' % (i, i,c))
        plt.close()


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_pairplots_df_v2(df0,idx_target=0):

    columns = df0.columns.to_numpy()
    target = columns[idx_target]
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    add_noise = True
    transparency = 0.95

    for i in range(len(idx)-1):
        for j in range(i+1,len(idx)):
            c1, c2 = columns[idx[i]], columns[idx[j]]
            df = df0[[target,c1,c2]]
            df = df.dropna()
            df = tools_DF.hash_categoricals(df)

            plt.clf()
            tools_plot_v2.plot_2D_features_pos_neg(df[[c1, c2]].to_numpy(), df[target].to_numpy(), xlabel=c1,ylabel=c2,add_noice=add_noise,transparency=transparency,filename_out=folder_out + 'pairplot_%02d_%02d_%s_%s.png'%(i,j,c1,c2))

    for i in range(len(idx)):
        c1 = columns[idx[i]]
        df = df0[[target, c1]]
        df = df.dropna()
        df = tools_DF.hash_categoricals(df)
        plt.clf()
        tools_plot_v2.plot_1D_features_pos_neg(plt, df[[c1]].to_numpy(), df[target].to_numpy(), labels=c1, filename_out=folder_out + 'pairplot_%02d_%02d_%s.png' % (i, i,c1))


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    ex_pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():
    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    ex_pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic():
    df = seaborn.load_dataset('titanic')

    #survived
    ex_pairplots_df(df, idx_target=0)
    #ex_pairplots_df_v2(df, idx_target=0)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,create=True)

    ex_titatic()
