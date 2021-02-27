import numpy
import pandas as pd
from sklearn.datasets import make_regression
# ----------------------------------------------------------------------------------------------------------------------
import classifier_KNN
#import classifier_SVM
import classifier_LM
import classifier_DTree
import classifier_RF
import classifier_Ada
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_moon():
    C = classifier_KNN.classifier_KNN()
    df,target = pd.read_csv(folder_in+'dataset_moons.csv', sep='\t'),0
    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=target,idx_columns=[1,2])
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    C = classifier_KNN.classifier_KNN()
    df,target = pd.read_csv(folder_in+'dataset_heart.csv', sep=','),-1
    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=target)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():
    #C = classifier_KNN.classifier_KNN()
    #C = classifier_DTree.classifier_DT(folder_out=folder_out)
    #C = classifier_RF.classifier_RF()
    #C = classifier_Ada.classifier_Ada()
    C = classifier_LM.classifier_LM()

    df,idx_target = pd.read_csv(folder_in+'dataset_titanic.csv', sep='\t'),0
    df.drop(labels=['alive', 'deck'], axis=1, inplace=True)

    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=idx_target)
    tools_plot_v2.pairplots_df(df, idx_target=idx_target, folder_out=folder_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_random():
    X, Y = make_regression(n_samples=1250, n_features=4, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1

    C = classifier_LM.classifier_LM()
    df = pd.DataFrame(data=(numpy.hstack((Y.reshape((-1, 1)), X))),columns=['target'] + ['%d' % c for c in range(X.shape[1])])
    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=0)
    tools_plot_v2.pairplots_df(df, idx_target=0, folder_out=folder_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    #ex_titanic()
    ex_random()

