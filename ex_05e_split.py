from sklearn.model_selection import KFold, ShuffleSplit,StratifiedKFold, GroupShuffleSplit,GroupKFold, StratifiedShuffleSplit
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def visualize_splits(cross_val, X_features, Y_target, group):

    n_splits = cross_val.n_splits
    grid_1d = numpy.arange(0, Y_target.shape[0], 1)
    X,Y = [],[]
    for split_number in range(1+n_splits):
        for g,y in zip(grid_1d,Y_target):
            X.append((g,split_number))
            Y.append(y)

    X = numpy.array(X,dtype=numpy.float32).reshape((1+n_splits,-1,2))
    Y = numpy.array(Y).reshape((1+n_splits,-1))
    Y_train_test = Y.copy()
    Y_group = Y.copy()

    for split_number, (idx_train, idx_test) in enumerate(cross_val.split(X=X_features, y=Y_target, groups=group)):
        Y[split_number, :] = Y_target[numpy.concatenate((idx_train, idx_test))]
        Y_train_test[split_number,idx_train] = 1
        Y_train_test[split_number,idx_test ] = 0
        Y_group[split_number,idx_train] = group[idx_train]
        Y_group[split_number,idx_test] = group[idx_test]
        X[split_number,idx_train,1]+= 0.1
        X[split_number,idx_test, 1]-= 0.1


    Y[-1, :] = Y_target
    Y_group[-1, :] = group
    Y_train_test[-1] = 0

    P.plot_2D_features_multi_Y(X.reshape((-1,2)),  Y.flatten()           ,palette='tab10', filename_out='splitter_%s_target.png' % (type(cross_val).__name__))
    P.plot_2D_features_multi_Y(X.reshape((-1, 2)), Y_group.flatten()     ,palette='Dark2', filename_out='splitter_%s_group.png' % (type(cross_val).__name__))

    return
# ----------------------------------------------------------------------------------------------------------------------
def get_data(n_points,n_groups,shuffle_target=False):
    X = numpy.random.randn(n_points, 1)
    #percentiles_classes = [.1, .3, .6]
    percentiles_classes = [0.4, 0.6]
    Y = numpy.hstack([[ii] * int(n_points * perc)for ii, perc in enumerate(percentiles_classes)])
    if shuffle_target:
        Y = Y[numpy.random.choice(Y.shape[0],Y.shape[0],replace=False)]

    groups = numpy.linspace(0,n_groups,n_points+1).astype(numpy.int)[:n_points]
    return X,Y,groups
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    n_points = 200
    n_groups = 3
    n_splits = 3
    X_features, Y_target, groups = get_data(n_points,n_groups)

    for cross_val in [KFold, ShuffleSplit, StratifiedKFold, GroupShuffleSplit, StratifiedShuffleSplit,GroupKFold]:
        visualize_splits(cross_val(n_splits=n_splits), X_features, Y_target, groups)


