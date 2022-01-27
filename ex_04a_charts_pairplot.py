import numpy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=False)
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    P.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():
    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    P.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic():
    df = seaborn.load_dataset('titanic')

    # survived
    idx_target = 0
    palette = 'tab10'


    #
    # idx_target = 3
    # target_min = df.iloc[:, idx_target].min()
    # target_max = df.iloc[:, idx_target].max()
    # N = 10
    # bins = numpy.linspace(target_min, target_max, N)
    # XX = pd.cut(df.iloc[:, idx_target], bins=bins,labels=bins[:-1].astype(int)).values.to_numpy()
    # df.iloc[:, idx_target] = XX
    # palette = plt.get_cmap('jet')
    # P.pairplots_df(df, idx_target,add_noise=True, palette=palette,mode2d=True)

    P.jointplots_df(df, idx_target)
    P.pairplots_df(df, idx_target, cumul_mode=True)
    #P.pairplots_df(df, idx_target, add_noise=False,palette=palette)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,create=True)
    ex_titatic()







