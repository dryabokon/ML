import numpy
import pandas as pd
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score, confusion_matrix
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v3(n_samples=250, n_features = 3):

    X, Y = make_regression(n_samples=n_samples, n_features=n_features,noise=50.0)
    Y[Y<=0]=0
    Y[Y >0]=1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_out_of_box(X_train, Y_train, X_test, Y_test):
    regr = linear_model.LogisticRegression(solver='liblinear')
    regr.fit(X_train, Y_train)
    Y_pred_train = regr.predict(X_train).flatten()
    Y_pred_test  = regr.predict(X_test).flatten()
    Y_pred_prob_train = regr.predict_proba(X_train)
    Y_pred_prob_test = regr.predict_proba(X_test)



    print('Method       \tTrain\tTest\n' + '-' * 30)
    print('Cross H Loss:\t%1.4f\t%1.4f'%(log_loss(Y_train, Y_pred_prob_train),log_loss(Y_test, Y_pred_prob_test)))
    print('F1 score    :\t%1.4f\t%1.4f' % (f1_score(Y_train, Y_pred_train), f1_score(Y_test, Y_pred_test)))
    print('confusion_matrix train')
    print(confusion_matrix(Y_train, Y_pred_train))
    print('confusion_matrix test')
    print(confusion_matrix(Y_test, Y_pred_test))

    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_in_house(X_train, Y_train, X_test, Y_test):
    regr = linear_model.LogisticRegression(solver='liblinear')
    regr.fit(X_train, Y_train)
    Y_pred_prob_train = 1/(1+numpy.exp(-(numpy.dot(X_train, regr.coef_.T) + regr.intercept_)))
    Y_pred_prob_test  = 1/(1+numpy.exp(-(numpy.dot(X_test , regr.coef_.T) + regr.intercept_)))
    Y_pred_train = 1*(Y_pred_prob_train>0.5)
    Y_pred_test = 1*(Y_pred_prob_test>0.5)

    #print('Method       \tTrain\tTest\n' + '-' * 30)
    print('Cross H Loss:\t%1.4f\t%1.4f' % (log_loss(Y_train, Y_pred_prob_train), log_loss(Y_test, Y_pred_prob_test)))
    print('F1 score    :\t%1.4f\t%1.4f' % (f1_score(Y_train, Y_pred_train), f1_score(Y_test, Y_pred_test)))
    print('confusion_matrix train')
    print(confusion_matrix(Y_train, Y_pred_train))
    print('confusion_matrix test')
    print(confusion_matrix(Y_test, Y_pred_test))

    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = get_data_v3()
    ex_regression_out_of_box(X_train, Y_train, X_test, Y_test)
    ex_regression_in_house(X_train, Y_train, X_test, Y_test)

    df = pd.DataFrame(data=(numpy.hstack((Y_train.reshape((-1,1)),X_train))),columns=['target']+['%d'%c for c in range(X_train.shape[1])])
    tools_plot_v2.pairplots_df(df,idx_target=0,folder_out=folder_out)
