import numpy as numpy
import math
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn import metrics
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v1(filename_in,idx_target=0):
    df = pd.read_csv(filename_in, sep='\t')

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    df_train, df_test = train_test_split(df.dropna(), test_size=0.5, shuffle=True)
    X_train, Y_train = df_train.iloc[:, idx].to_numpy(), df_train.iloc[:, [idx_target]].to_numpy()
    X_test , Y_test  = df_test.iloc[:, idx].to_numpy(), df_test.iloc[:, [idx_target]].to_numpy()

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v3(n_samples=250, n_features = 3):

    X, Y = make_regression(n_samples=n_samples, n_features=n_features,noise=50.0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

    Y_train += numpy.random.random(Y_train.shape)
    Y_test += numpy.random.random(Y_test.shape)

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_out_of_box(X_train, Y_train, X_test, Y_test):

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_train_pred = regr.predict(X_train)
    Y_test_pred = regr.predict(X_test)

    r2a_train = r2_score(Y_train_pred, Y_train)
    r2a_test = r2_score(Y_test_pred, Y_test)

    P.plot_fact_predict(Y_test, Y_test_pred, filename_out='P.png')

    print('Method  \tTrain\tTest\n' + '-' * 30)
    print('MSE linrg:\t%1.4f\t%1.4f' % (math.sqrt(((Y_train - Y_train_pred) ** 2).mean()),math.sqrt(((Y_test - Y_test_pred) ** 2).mean())))
    print('R2  linrg:\t%1.4f\t%1.4f' % (r2a_train,r2a_test))
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_in_house(X_train, Y_train, X_test, Y_test):

    X1_train = numpy.hstack((X_train,numpy.full((X_train.shape[0],1),1)))
    X1_test  = numpy.hstack((X_test, numpy.full((X_test.shape[0], 1), 1)))
    a2 = numpy.matmul(numpy.linalg.inv(numpy.matmul(X1_train.T,X1_train)),X1_train.T).dot(Y_train).reshape((-1,1))
    Y_train_pred = (numpy.matmul(X1_train, a2)).flatten()
    Y_test_pred = (numpy.matmul(X1_test,a2)).flatten()

    r2_train = metrics.r2_score(Y_train_pred, Y_train)
    r2_test  = metrics.r2_score(Y_test_pred, Y_test)

    #print('Method  \tTrain\tTest\n' + '-' * 30)
    print('MSE linrg:\t%1.4f\t%1.4f' % (math.sqrt(((Y_train - Y_train_pred) ** 2).mean()),math.sqrt(((Y_test - Y_test_pred) ** 2).mean())))
    print('R2  linrg:\t%1.4f\t%1.4f' % (r2_train,r2_test))
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = get_data_v3()

    ex_regression_out_of_box(X_train, Y_train, X_test, Y_test)
    ex_regression_in_house  (X_train, Y_train, X_test, Y_test)