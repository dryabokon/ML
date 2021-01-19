import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_plot_df(df, idx_target, idx_num, idx_cat, filename_out):
    columns = df.columns.to_numpy()
    name_num, name_cat, name_target  = columns[[idx_num, idx_cat, idx_target]]
    seaborn.lmplot(x=name_num, y=name_target, col=name_cat, hue=name_cat, data=df, y_jitter=.02, logistic=True,truncate=False)
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = seaborn.load_dataset('titanic')
    ex_regression_plot_df(df, 0, 3, 1, folder_out + 'pairplot.png')


