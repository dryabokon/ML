import numpy
import pandas as pd
from sklearn.datasets import make_classification
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_RF,classifier_LM
import tools_ML_v2
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
#C = classifier_SVM.classifier_SVM()
#C = classifier_RF.classifier_RF()
C = classifier_LM.classifier_LM()
ML = tools_ML_v2.ML(C, folder_out=folder_out)
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_random():
    X, Y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
    idx_target = 0
    df = pd.DataFrame(data=(numpy.hstack((Y.reshape((-1, 1)), X))),columns=['target'] + ['%d' % c for c in range(X.shape[1])])
    df.iloc[:,idx_target] = df.iloc[:,idx_target].astype(int)
    ML.P.set_color(0, P.color_blue)
    ML.P.set_color(1, P.color_red)
    ML.P.set_color(2, P.color_gray)

    df_metrics = ML.E2E_train_test_df(df, idx_target=idx_target, do_charts=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    # model = LogisticRegression(multi_class='ovr')
    # model.fit(X, y)
    # yhat = model.predict(X)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_iris():
    df,idx_target = pd.read_csv(folder_in + 'dataset_iris.csv'),0
    df.iloc[:,0] = df.iloc[:,0].map({0:'type A',1:'type B',2:'type C'})
    ML.P.set_color('type A', P.color_amber)
    ML.P.set_color('type B', P.color_gold)
    ML.P.set_color('type C', P.color_blue)

    df_metrics = ML.E2E_train_test_df(df,idx_target=0,do_charts=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    #ML.P.pairplots_df(df, idx_target=idx_target,cumul_mode=False,add_noise=False)
    #P.histoplots_df(df, idx_target=idx_target,transparency=0.75)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_glass():
    df = pd.read_csv(folder_in + 'dataset_glass.csv')
    df = df.drop(columns=['Id number'])
    idx_target = df.columns.get_loc('Type of glass')
    df_metrics = ML.E2E_train_test_df(df,idx_target=idx_target,do_charts=True)
    print(tools_DF.prettify(df_metrics, showindex=False))

    P.pairplots_df(df, idx_target=idx_target,cumul_mode=False,add_noise=False)
    P.histoplots_df(df, idx_target=idx_target,transparency=0.75)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #ex_random()
    #ex_iris()
    ex_glass()



