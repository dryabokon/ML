import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_importance_df(df,idx_target):

    columns = df.columns.to_numpy()
    col_types = numpy.array([str(t) for t in df.dtypes])
    are_categoirical = numpy.array([cc in ['object', 'category', 'bool'] for cc in col_types])

    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    are_categoirical = numpy.delete(are_categoirical, idx_target)
    idx = idx[~are_categoirical]

    X = df.iloc[:,idx].to_numpy()
    Y = df.iloc[:,[idx_target]].to_numpy().flatten()

    tools_plot.plot_feature_importance(plt, plt.figure(), X=X, Y=Y,header=columns[idx], filename_out = folder_out + 'FI.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = pd.read_csv(folder_in + 'dataset_wine.txt', sep='\t')
    ex_feature_importance_df(df,idx_target=0)