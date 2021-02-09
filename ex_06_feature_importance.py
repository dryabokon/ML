import numpy
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def select_features(X, Y):

    fs = SelectKBest(score_func=f_regression, k='all')

    for c in range(X.shape[1]):
        min_value = X[:,c].min()
        if min_value < 0:
            X[:, c]+=-min_value

    fs.fit(X, Y)
    return fs.scores_/fs.scores_.sum()
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_LM(df, idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    feature_names = columns[idx]
    X = df.iloc[:,idx].to_numpy()
    Y = df.iloc[:,idx_target].to_numpy()

    f_scores = select_features(X, Y)
    f_scores = f_scores/f_scores.sum()

    idx = numpy.argsort(-f_scores)
    print('\nscore\tfeature\n--------------')
    for feature_name,f_score in zip(feature_names[idx], f_scores[idx]):
        print('%1.2f\t%s'%(f_score,feature_name))

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_LM_v2(df, idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    columns = df.columns.to_numpy()
    regr = linear_model.LinearRegression()
    idx = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)
    feature_names = columns[idx]
    X = df.iloc[:, idx].to_numpy()
    Y = df.iloc[:, idx_target].to_numpy()
    regr.fit(X, Y)
    A = regr.coef_.flatten()
    stdevs = numpy.array([df[c].std() for c in columns[idx]])

    values = numpy.abs(A)*stdevs
    values = values/values.sum()

    idx = numpy.argsort(-values)
    print('\nscore\tfeature\n--------------')
    for feature_name, f_score in zip(feature_names[idx], values[idx]):
        print('%1.2f\t%s' % (f_score, feature_name))

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_importance_df(df,idx_target):

    df=tools_DF.hash_categoricals(df)

    X,Y = tools_DF.df_to_XY(df,idx_target,keep_categoirical=False)
    feature_names = tools_DF.get_names(df,idx_target,keep_categoirical=False)
    tools_plot_v2.plot_feature_importance_LM(X, Y, feature_names, filename_out=folder_out + 'FI_LM.png')
    tools_plot_v2.plot_feature_importance_XGB(X, Y, feature_names, filename_out = folder_out + 'FI_XGB.png')


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_carts(df, idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    columns = df.columns.to_numpy()
    regr = linear_model.LinearRegression()
    idx = numpy.delete(numpy.arange(0, df.shape[1]), idx_target)
    feature_names = columns[idx]
    X = df.iloc[:, idx].to_numpy()
    Y = df.iloc[:, idx_target].to_numpy()

    for feature_name,x in zip(feature_names,X.T):
        tools_plot_v2.plot_regression_YX(Y, x, logistic=False, filename_out=folder_out+'%s.png'%feature_name)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #df,idx_target = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=','),2
    df,idx_target = pd.read_csv(folder_in + 'dataset_titanic.txt', delimiter='\t'),0

    ex_feature_imporance_LM(df, idx_target)
    #ex_feature_imporance_carts(df, idx_target)
    ex_feature_importance_df(df,idx_target)