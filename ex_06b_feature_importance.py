import numpy
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, f_classif, mutual_info_classif
from sklearn import linear_model
from sklearn.metrics import r2_score
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def select_features(X, Y):
    #score_func = mutual_info_classif
    score_func = f_regression
    fs = SelectKBest(score_func=score_func, k='all')

    for c in range(X.shape[1]):
        min_value = X[:,c].min()
        if min_value < 0:
            X[:, c]+=-min_value

    fs.fit(X, Y)
    xxx = fs.scores_
    norm = fs.scores_.sum()
    result =  xxx/norm

    return result
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_F_score(df, idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    X = df.iloc[:,idx].to_numpy()
    Y = df.iloc[:,idx_target].to_numpy()

    f_scores = select_features(X, Y)
    f_scores = f_scores/f_scores.sum()

    return f_scores
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_C(df, idx_target=0):
    #https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    X = df.iloc[:,idx].to_numpy()
    Y = df.iloc[:,idx_target].to_numpy()
    ridgereg = linear_model.Ridge(alpha=0.001, normalize=True)
    ridgereg.fit(X, Y)

    values = numpy.abs(ridgereg.coef_ * X.std(axis=0))
    return  values
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_imporance_R2(df, idx_target=0):

    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    regr = linear_model.LinearRegression()
    Y = df.iloc[:, idx_target].to_numpy()

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    X = df.iloc[:, idx].to_numpy()

    R2s = []
    for i in range(X.shape[1]):
        idx = numpy.delete(numpy.arange(0, X.shape[1]), i)
        x = X[:,idx]
        regr.fit(x, Y)
        Y_pred = regr.predict(x).flatten()
        R2s.append(100*r2_score(Y, Y_pred))

    return numpy.array(R2s)
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_feature_imporance(df,idx_target):
    columns = df.columns.to_numpy()[numpy.delete(numpy.arange(0, df.shape[1]), idx_target)]
    S1 = ex_feature_imporance_F_score(df, idx_target)
    S2 = ex_feature_imporance_R2(df, idx_target)
    S3 = ex_feature_imporance_C(df, idx_target)

    idx = numpy.argsort(S2)
    print('\nscore\texclR2\tC     \tfeature\n-------------------------')
    for feature_name, s1, s2, s3 in zip(columns[idx], S1[idx], S2[idx], S3[idx]):
        print('%1.2f\t%1.2f\t%1.2f\t%s' % (s1, s2, s3, feature_name))
    return

# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():

    df,idx_target = pd.read_csv(folder_in + 'dataset_titanic.txt', delimiter='\t'),0
    df.drop(labels = ['alive'], axis = 1, inplace = True)
    evaluate_feature_imporance(df, idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():

    df,idx_target = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=','),2
    evaluate_feature_imporance(df, idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_fuel():
    df,idx_target = pd.read_csv(folder_in + 'dataset_fuel.csv', delimiter='\t'),0
    df.drop(labels=['name'], axis=1, inplace=True)
    evaluate_feature_imporance(df, idx_target)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_fuel()