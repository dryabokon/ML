import numpy
import seaborn
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

def ex_heart():
    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    tools_plot_v2.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():
    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    tools_plot_v2.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic():
    df = seaborn.load_dataset('titanic')

    #survived
    tools_plot_v2.pairplots_df(df, idx_target=0)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,create=True)

    ex_titatic()
