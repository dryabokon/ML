import matplotlib.pyplot as plt
import numpy as numpy
import math
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
# ----------------------------------------------------------------------------------------------------------------------
def get_data():
    df = pd.read_csv(folder_in+'dataset_diabetes.txt', sep='\t')
    df_train, df_test = train_test_split(df.dropna(), test_size=0.5, shuffle=True)
    X_train, Y_train = df_train.iloc[:, 1:].to_numpy(), df_test.iloc[:, 0].to_numpy()
    X_test, Y_test = df_train.iloc[:, 1:].to_numpy(), df_test.iloc[:, 0].to_numpy()

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def get_data2():
    N=1000
    M = 3

    X1 = numpy.random.random((N,M))
    X1 = numpy.hstack((X1, numpy.full((X1.shape[0], 1), 1)))
    a = numpy.random.random((M+1,1))
    Y = numpy.matmul(X1, a)

    idx_train = numpy.arange(0,N//2,1)
    idx_test = numpy.arange(N//2, N,1)

    X_train, Y_train = X1[idx_train,:-1],Y[idx_train]
    X_test, Y_test= X1[idx_test, :-1], Y[idx_test]

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = get_data2()

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    loss =  math.sqrt(((Y_pred - Y_test)**2).mean())
    print('LinReg loss = %f\n'%loss)


    X1_train = numpy.hstack((X_train,numpy.full((X_train.shape[0],1),1)))
    X1_test  = numpy.hstack((X_test, numpy.full((X_test.shape[0], 1), 1)))
    a = numpy.matmul(numpy.linalg.inv(numpy.matmul(X1_train.T,X1_train)),X1_train.T).dot(Y_train).reshape((-1,1))
    Y_pred = numpy.matmul(X1_test,a)
    loss = math.sqrt(((Y_pred - Y_test) ** 2).mean())
    print('Custom loss = %f'%loss)

    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_PCA_train = pca.transform(X_train)
    X_PCA_test = pca.transform(X_test)

    X1_PCA_train = numpy.hstack((X_PCA_train, numpy.full((X_PCA_train.shape[0], 1), 1)))
    X1_PCA_test = numpy.hstack((X_PCA_test, numpy.full((X_PCA_test.shape[0], 1), 1)))
    a_pca = numpy.matmul(numpy.linalg.inv(numpy.matmul(X1_PCA_train.T,X1_PCA_train)),X1_PCA_train.T).dot(Y_train).reshape((-1,1))
    Y_pred = numpy.matmul(X1_PCA_test, a_pca)
    loss_test = math.sqrt(((Y_pred - Y_test) ** 2).mean())
    print('PCA loss = %f'%loss_test)
