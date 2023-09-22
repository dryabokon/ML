import numpy
import pandas as pd
from sklearn.datasets import make_regression
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_KNN
from classifier import classifier_SVM
from classifier import classifier_LM
from classifier import classifier_DTree
#from classifier import classifier_RF
from classifier import classifier_Ada
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
import tools_plot_v2
import tools_DF
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
# C = classifier_KNN.classifier_KNN()
# C = classifier_DTree.classifier_DT()
# C = classifier_RF.classifier_RF()
# C = classifier_Ada.classifier_Ada()
# C = classifier_KNN.classifier_KNN()
C = classifier_LM.classifier_LM()
P = tools_plot_v2.Plotter(folder_out)
ML = tools_ML_v2.ML(C, folder_out=folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_moon():

    df,idx_target = pd.read_csv(folder_in+'dataset_moons.csv', sep='\t'),0
    df.iloc[:,idx_target]=df.iloc[:,idx_target].astype(int)
    df_metrics = ML.E2E_train_test_df(df,idx_target=idx_target,idx_columns=[1,2],do_charts=True,do_pca=True)
    print(tools_DF.prettify(df_metrics,showindex=False))

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_red)
    P.histoplots_df(df, idx_target=idx_target)
    P.pairplots_df(df, idx_target=idx_target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    df,idx_target = pd.read_csv(folder_in+'dataset_heart.csv', sep=','),-1
    df_metrics = ML.E2E_train_test_df(df,idx_target=idx_target,do_charts=True,do_pca=True)
    print(tools_DF.prettify(df_metrics,showindex=False))

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_red)
    P.histoplots_df(df, idx_target=idx_target)
    P.pairplots_df(df, idx_target=idx_target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_random():
    X, Y = make_regression(n_samples=1250, n_features=3, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1

    df = pd.DataFrame(numpy.concatenate([X * 100, Y.reshape((-1, 1))], axis=1),columns=['C%d' % c for c in range(X.shape[1])] + ['target'])
    idx_target = df.columns.get_loc('target')
    df_metrics = ML.E2E_train_test_df(df, idx_target=idx_target,do_charts=True,do_pca=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_red)
    P.histoplots_df(df, idx_target=idx_target)
    P.pairplots_df(df, idx_target=idx_target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_flights_kibana():

    df,idx_target = pd.read_csv(folder_in+'dataset_kibana_flights.csv', sep=','),10
    df = df.drop(labels=['_source.FlightDelayMin','_source.FlightDelayType'], axis=1)
    df = tools_DF.hash_categoricals(df)
    df_metrics = ML.E2E_train_test_df(df,idx_target=idx_target,do_charts=True,do_pca=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_red)
    P.histoplots_df(df, idx_target=idx_target)
    P.pairplots_df(df, idx_target=idx_target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():

    df = pd.read_csv(folder_in+'dataset_titanic.csv', sep='\t')
    df.drop(columns=['alive','deck'],inplace=True)
    idx_target = df.columns.get_loc('survived')
    df_metrics = ML.E2E_train_test_df(df, idx_target=idx_target, do_charts=True, do_pca=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_amber)
    P.histoplots_df(df, idx_target=idx_target)
    #df = tools_DF.hash_categoricals(df)
    P.pairplots_df(df, idx_target=idx_target,cumul_mode=False,add_noise=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out)
    #ex_moon()
    #ex_heart()
    ex_random()
    #ex_flights_kibana()
    #ex_titanic()