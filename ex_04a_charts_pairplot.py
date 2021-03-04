import seaborn
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_IO
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
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

    #P.pairplots_df(df, idx_target)
    #P.pairplots_df_v2(df, idx_target, add_noise=False)
    P.pairplots_df_v2(df, idx_target, add_noise=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,create=True)

    ex_titatic()



