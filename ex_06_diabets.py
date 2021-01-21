import numpy
import pandas as pd
import math
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    array = dataframe.values

    X_train = array[:, 0:8]
    Y_train = array[:, 8]

    selector = SelectKBest(score_func=chi2, k=4)
    selector = selector.fit(X_train, Y_train)
    #features = selector.transform(X_train)

    importance = selector.scores_
    idx = numpy.argsort(-importance)[:3]
    features = X_train[:,idx]


    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred1 = regr.predict(X_train).flatten()
    print('Original loss = %f\n' % math.sqrt(((Y_pred1 - Y_train) ** 2).mean()))


    regr.fit(features, Y_train)
    Y_pred2 = regr.predict(features).flatten()
    print('Features loss = %f\n' % math.sqrt(((Y_pred2 - Y_train) ** 2).mean()))