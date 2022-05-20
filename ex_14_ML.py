import numpy
import pandas as pd
from sklearn.datasets import make_regression
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_KNN
from classifier import classifier_SVM
from classifier import classifier_LM
from classifier import classifier_DTree
from classifier import classifier_RF
from classifier import classifier_Ada
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_moon():
    C = classifier_KNN.classifier_KNN()
    df,target = pd.read_csv(folder_in+'dataset_moons.csv', sep='\t'),0
    ML = tools_ML_v2.ML(C)
    ML.E2E_train_test_df(df,idx_target=target,idx_columns=[1,2])
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    C = classifier_KNN.classifier_KNN()
    df,target = pd.read_csv(folder_in+'dataset_heart.csv', sep=','),-1
    ML = tools_ML_v2.ML(C)
    ML.E2E_train_test_df(df,idx_target=target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():
    #C = classifier_KNN.classifier_KNN()
    #C = classifier_DTree.classifier_DT(folder_out=folder_out)
    #C = classifier_RF.classifier_RF()
    #C = classifier_Ada.classifier_Ada()
    #C = classifier_LM.classifier_LM()
    C = classifier_SVM.classifier_SVM()
    P = tools_plot_v2.Plotter(folder_out)

    df0 = pd.read_csv(folder_in+'dataset_titanic.csv', sep='\t')
    df0.drop(columns=['alive','deck'],inplace=True)

    df = df0.copy()
    df=df[['sex','who','survived']]
    df = tools_DF.hash_categoricals(df)

    #ML = tools_ML_v2.ML(C,folder_out)
    #ML.E2E_train_test_df(df,idx_target=df.columns.get_loc('survived'),do_charts=True,do_density=True,do_pca = True)
    #P.pairplots_df(df, idx_target=df.columns.get_loc('survived'),marker_size=6,transparency=0.90,add_noise=True)

    P.set_color(0,P.color_blue)
    P.set_color(1,P.color_red)
    P.histoplots_df(df0, idx_target=df0.columns.get_loc('survived'))

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_flights_kibana():
    #C = classifier_KNN.classifier_KNN()
    #C = classifier_DTree.classifier_DT(folder_out=folder_out)
    #C = classifier_RF.classifier_RF()
    #C = classifier_Ada.classifier_Ada()
    C = classifier_LM.classifier_LM()
    P = tools_plot_v2.Plotter(folder_out)

    df,idx_target = pd.read_csv(folder_in+'dataset_kibana_flights.csv', sep=','),10
    df.drop(labels=['_source.FlightDelayMin','_source.FlightDelayType'], axis=1, inplace=True)

    ML = tools_ML_v2.ML(C,folder_out)
    ML.E2E_train_test_df(df,idx_target=idx_target)
    #P.pairplots_df(df, idx_target=idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_random():
    X, Y = make_regression(n_samples=1250, n_features=3, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1

    C = classifier_LM.classifier_LM()
    P = tools_plot_v2.Plotter(folder_out)
    df = pd.DataFrame(data=(numpy.hstack((Y.reshape((-1, 1)), X))),columns=['target'] + ['%d' % c for c in range(X.shape[1])])
    P.pairplots_df(df, idx_target=0)

    ML = tools_ML_v2.ML(C, folder_out)
    ML.E2E_train_test_df(df,idx_target=0)


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_titanic()
    #ex_random()
    #ex_flights_kibana()

