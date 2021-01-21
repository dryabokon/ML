import numpy as numpy
import math
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import chi2
from skfeature.function.similarity_based import fisher_score
from scipy.sparse import *
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def get_feature_names(filename_in,idx_target=0):
    df = pd.read_csv(filename_in, sep='\t')
    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    return columns[idx]
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v1(filename_in,idx_target=0,force_positive=False):
    df = pd.read_csv(filename_in, sep='\t')


    if force_positive:
        for c in df.columns:
            a = df.loc[:, c].min()
            if a<0:
                df.loc[:,c]+=-a


    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    df_train, df_test = train_test_split(df.dropna(), test_size=0.5, shuffle=True)
    X_train, Y_train = df_train.iloc[:, idx].to_numpy(), df_train.iloc[:, [idx_target]].to_numpy()
    X_test , Y_test  = df_test.iloc[:, idx].to_numpy(), df_test.iloc[:, [idx_target]].to_numpy()



    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v2(n_samples=50, n_features = 3):

    X1 = numpy.hstack((numpy.random.random((n_samples, n_features)), numpy.full((n_samples, 1), 1)))
    a = -1+2*numpy.random.random((n_features + 1, 1))
    a[0]=0

    Y = numpy.matmul(X1, a)
    idx_train = numpy.arange(0, n_samples // 2, 1)
    idx_test = numpy.arange(n_samples // 2, n_samples, 1)
    X_train, Y_train = X1[idx_train,:-1], Y[idx_train]
    X_test , Y_test  = X1[idx_test, :-1], Y[idx_test]

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def get_data_v3(n_samples=50, n_features = 3):

    X, Y = make_regression(n_samples=n_samples, n_features=n_features)

    idx_train = numpy.arange(0, n_samples // 2, 1)
    idx_test = numpy.arange(n_samples // 2, n_samples, 1)
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_test, Y_test = X[idx_test], Y[idx_test]

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def get_stats(X,Y,a,header):

    sigma_sq = 1#numpy.std(Y)

    I = numpy.sqrt(numpy.linalg.inv(numpy.matmul(X1_train.T, X1_train)))

    for i in range(X.shape[1]):
        Z_score = a[i]/(sigma_sq*I[i,i])
        print('%1.2f\t%1.2f\t%1.2f\t%s'%(a[i],I[i,i],Z_score,header[i]))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    feature_names = get_feature_names(folder_in+'dataset_diabetes.txt')

    #X_train, Y_train, X_test, Y_test = get_data_v1(folder_in+'dataset_diabetes.txt',force_positive=True)
    X_train, Y_train, X_test, Y_test = get_data_v1(folder_in+'dataset_cancer.txt',idx_target=8,force_positive=True)
    #X_train, Y_train, X_test, Y_test = get_data_v3()

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred1 = regr.predict(X_test).flatten()
    print('LinReg loss = %f\n'%math.sqrt(((Y_pred1 - Y_test)**2).mean()))
    a1 = numpy.hstack((regr.coef_.flatten().reshape((1,-1)),numpy.array(regr.intercept_).reshape(1,1)))

    X1_train = numpy.hstack((X_train,numpy.full((X_train.shape[0],1),1)))
    X1_test  = numpy.hstack((X_test, numpy.full((X_test.shape[0], 1), 1)))
    a2 = numpy.matmul(numpy.linalg.inv(numpy.matmul(X1_train.T,X1_train)),X1_train.T).dot(Y_train).reshape((-1,1))
    Y_pred2 = (numpy.matmul(X1_test,a2)).flatten()
    print('Custom loss = %f'%math.sqrt(((Y_pred2 - Y_test) ** 2).mean()))

    print('Coefficient of determination: %.2f'% r2_score(Y_pred2, Y_test))

    f_scores = chi2(X_train, Y_train)


    for feature_name, f_score in zip(feature_names, f_scores[1]):
        print('%s\t%1.2f' % (feature_name, f_score))

    # tools_IO.remove_files(folder_out)
    # for c,x in enumerate(X_train.T):
    #     tools_plot_v2.plot_regression_YX(Y_train, x,logistic=False,filename_out=folder_out+'r_%02d.png'%c)