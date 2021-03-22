import seaborn as sns
import numpy
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.datasets import make_regression, make_classification
from matplotlib import colors
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_LM
import tools_ML_v2
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def get_SMOTE(X, Y,do_debug=False):
    X_Sampled, Y_Sampled = SMOTE().fit_resample(X, Y)
    if do_debug:
        x_range = numpy.array([X_Sampled[:, 0].min(), X_Sampled[:, 0].max()])
        y_range = numpy.array([X_Sampled[:, 1].min(), X_Sampled[:, 1].max()])
        df_sampled = pd.DataFrame(numpy.concatenate((Y_Sampled.reshape(-1, 1), X_Sampled), axis=1), columns=['Y', 'x0', 'x1'])
        df         = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X), axis=1),columns=['Y', 'x0', 'x1'])

        customPalette = ['#808080', '#C00000']

        P.plot_2D_features_v3(df, x_range=x_range,y_range=y_range,palette=customPalette,transparency=0.75,figsize=(6,4),filename_out='original.png')
        P.plot_2D_features_v3(df_sampled, x_range=x_range,y_range=y_range,palette=customPalette,transparency=0.75,figsize=(6,4), filename_out='SMOTE.png')
    return X_Sampled, Y_Sampled
# ----------------------------------------------------------------------------------------------------------------------
def get_SMOTE_UnderSampler(X, Y,do_debug=False):
    pipeline = Pipeline(steps=[('o', SMOTE(sampling_strategy=0.1)), ('u', RandomUnderSampler(sampling_strategy=0.5))])
    X_Sampled, Y_Sampled = pipeline.fit_resample(X, Y)
    if do_debug:
        x_range = numpy.array([X[:, 0].min(), X[:, 0].max()])
        y_range = numpy.array([X[:, 1].min(), X[:, 1].max()])
        df_sampled = pd.DataFrame(numpy.concatenate((Y_Sampled.reshape(-1, 1), X_Sampled), axis=1), columns=['Y', 'x0', 'x1'])
        df         = pd.DataFrame(numpy.concatenate((Y.reshape(-1, 1), X), axis=1),columns=['Y', 'x0', 'x1'])
        customPalette = ['#808080', '#C00000']
        P.plot_2D_features_v3(df, x_range=x_range,y_range=y_range,palette=customPalette,transparency=0.5,figsize=(6,4),filename_out='original.png')
        P.plot_2D_features_v3(df_sampled, x_range=x_range,y_range=y_range,palette=customPalette,transparency=0.5,figsize=(6,4),filename_out='SMOTE_UnderSampler.png')
    return X_Sampled, Y_Sampled
# ----------------------------------------------------------------------------------------------------------------------
def ex_search(X, Y):

    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
    #model = DecisionTreeClassifier()
    model = LogisticRegression()

    scores = cross_val_score(model, X, Y, scoring='roc_auc', cv=cv)
    print('k=%d, AUC: %.3f' % (0, scores.mean()))

    for k_neighbors in [1, 2, 3, 4, 5, 6, 7]:

        pipeline_pred = Pipeline(steps=[('over', SMOTE(sampling_strategy=0.1, k_neighbors=k_neighbors)),
                                   ('under', RandomUnderSampler(sampling_strategy=0.5)),
                                   ('model', model)])

        scores = cross_val_score(pipeline_pred, X, Y, scoring='roc_auc', cv=cv)
        print('k=%d, AUC: %.3f' % (k_neighbors, scores.mean()))


        # pipeline_sample = Pipeline(steps=[('over', SMOTE(sampling_strategy=0.1, k_neighbors=k_neighbors)),
        #                            ('under', RandomUnderSampler(sampling_strategy=0.5))])
        # X_sampled, Y_sampled= pipeline_sample.fit_resample(X, Y)
        # scores = cross_val_score(model, X_sampled, Y_sampled, scoring='roc_auc', cv=cv)
        # print('k=%d, AUC: %.3f' % (k_neighbors, scores.mean()))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_train_test(X,Y):
    C = classifier_LM.classifier_LM()
    ML = tools_ML_v2.ML(C, folder_out + 'original/')
    P = tools_plot_v2.Plotter(folder_out+'original/')

    df = pd.DataFrame(data=(numpy.hstack((Y.reshape((-1, 1)), X))),columns=['target'] + ['%d' % c for c in range(X.shape[1])])
    P.pairplots_df(df, idx_target=0)
    ML.E2E_train_test_df(df,idx_target=0)


    ML = tools_ML_v2.ML(C, folder_out + 'sampled/')
    P = tools_plot_v2.Plotter(folder_out + 'sampled/')
    X_Sampled, Y_Sampled = get_SMOTE_UnderSampler(X,Y,do_debug=True)
    df_sampled = pd.DataFrame(data=(numpy.hstack((Y_Sampled.reshape((-1, 1)), X_Sampled))),columns=['target'] + ['%d' % c for c in range(X.shape[1])])
    ML.E2E_train_test_df(df_sampled, idx_target=0)
    P.pairplots_df(df_sampled, idx_target=0)

    return
# ----------------------------------------------------------------------------------------------------------------------
def get_data():
    X, Y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],flip_y=0.01, random_state=1)
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v2():
    X, Y = make_regression(n_samples=10000, n_features=2, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1

    idx0 = numpy.where(Y<=0)[0]
    idx1 = numpy.where(Y>0)[0]
    idx1 = numpy.random.choice(len(idx1),int(len(idx1)*0.1))
    idx = numpy.concatenate((idx0,idx1))

    X = X[idx]
    Y = Y[idx]

    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X,Y = get_data()
    get_SMOTE(X, Y, do_debug=True)
    get_SMOTE_UnderSampler(X, Y, do_debug=True)

    #ex_train_test(X,Y)


