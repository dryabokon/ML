import numpy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_IO
import tools_Hyptest
import tools_feature_importance
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
HT = tools_Hyptest.HypTest()
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    P.pairplots_df(df, add_noise=True,idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():
    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    P.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic():
    df = seaborn.load_dataset('titanic')
    #df.to_csv(folder_out + 'titanic.csv', index=False)
    #df = tools_DF.impute_na(df,strategy='mean')
    df = df.drop(columns=['alive'])

    # survived
    idx_target = df.columns.get_loc('survived')

    df_Q = HT.distribution_distances(df, idx_target)
    df_Q.to_csv(folder_out + 'df_Q.csv')
    th = numpy.quantile(df_Q.values, 0.90)
    df_Q = (df_Q >= th)

    # sex
    # idx_target = df.columns.get_loc('sex')
    #df['sex'] = df['sex'].map({'male':0,'female':1})

    df['survived'] = df['survived'].map({0: '0-not survived', 1: '1-survived'})


    P.set_color('0-not survived',P.color_red)
    P.set_color('1-survived',P.color_blue)
    #P.histoplots_df(df, idx_target,transparency=0.5)
    P.pairplots_df(df, idx_target, df_Q, add_noise=True,remove_legend=False)


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,create=True)
    ex_titatic()








