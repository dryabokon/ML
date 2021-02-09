import numpy as numpy
import math
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
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
def get_data_v3(n_samples=50, n_features = 3):

    X, Y = make_regression(n_samples=n_samples, n_features=n_features)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

    Y_train += numpy.random.random(Y_train.shape)
    Y_test += numpy.random.random(Y_test.shape)

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = get_data_v3()

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
