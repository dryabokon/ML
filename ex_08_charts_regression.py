import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
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
def ex_regression_plot_YX(Y,X,filename_out):
    columns = ['Y','X']
    A = numpy.hstack((Y,X))
    df = pd.DataFrame(data=A, columns=columns)

    seaborn.lmplot(data=df, x=columns[1], y=columns[0], y_jitter=.02, logistic=True,truncate=False)
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    df = seaborn.load_dataset('titanic')
    #survived (age,pclass)
    # ex_regression_plot_df(df, 0, 3, 1, folder_out + 'pairplot.png')

    #survived (class, sex)
    #ex_regression_plot_df(df, 0, 1, 2, folder_out + 'pairplot.png')

    #survived (age, sex)
    ex_regression_plot_df(df, idx_target=0, idx_num=3, idx_cat=2, filename_out = folder_out + 'regression.png')



