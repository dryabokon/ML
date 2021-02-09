import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_pairplot_df(df,idx_target=0):

    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    target = columns[idx_target]
    idx =numpy.delete(numpy.arange(0,len(columns)),idx_target)
    vars = columns[idx]

    pal = numpy.array(['tab10', 'husl','Set2','Paired','hls'])[0]

    seaborn.pairplot(data=df, hue=target,vars=vars,palette=seaborn.color_palette(pal))
    plt.savefig(folder_out+'pairplot.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_pairplots_df(df,idx_target=0):

    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    target = columns[idx_target]
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    pal = numpy.array(['tab10', 'husl', 'Set2', 'Paired', 'hls'])[0]

    for i in range(len(idx)-1):
        for j in range(i+1,len(idx)):
            c1,c2 = columns[idx[i]], columns[idx[j]]
            plt.clf()
            seaborn.jointplot(data=df, x=c1, y=c2, hue=target,palette=pal)
            plt.savefig(folder_out + 'pairplot_%02d_%02d.png'%(i,j))
            plt.close()

    for i in range(len(idx)):
        c = columns[idx[i]]
        plt.clf()
        seaborn.histplot(data=df, x=c, hue=target, palette=pal,element='poly')
        plt.savefig(folder_out + 'pairplot_%02d_%02d.png' % (i, i))
        plt.close()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    ex_pairplots_df(df,idx_target=-1)

