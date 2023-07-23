import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
folder_in = './data/ex_datasets/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
T = tools_time_profiler.Time_Profiler()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    centers = [[1, 1], [-1, -1], [1, -1]]
    X, Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    P.plot_2D_features(pd.DataFrame({'Y': Y, 'X0': X[:, 0], 'X1': X[:, 1]}), filename_out='x.png')

    T.tic('DBSCAN')
    labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(X)
    T.print_duration('DBSCAN')
    P.plot_2D_features(pd.DataFrame({'Y': labels, 'X0': X[:, 0], 'X1': X[:, 1]}), filename_out='dbscan.png')


