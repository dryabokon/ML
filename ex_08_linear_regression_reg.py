import cv2
from sklearn.metrics import r2_score
import numpy as numpy
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV,LassoCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def prepare_data():
    df, idx_target = pd.read_csv(folder_in + 'dataset_crime.csv', delimiter='\t'), -1

    L = numpy.array(([1,2,3,126] + numpy.arange(101,118).tolist()+numpy.arange(121,125).tolist()))
    df.drop(df.columns.to_numpy()[L], axis=1, inplace=True)
    df.dropna(inplace=True)
    #cv2.imwrite(folder_out+'nans.png',255*(df.isnull()).to_numpy())

    y = df[df.columns.to_numpy()[-1]]
    X = df.drop(df.columns.to_numpy()[-1], axis=1)
    df.to_csv(folder_out+'dataset_crime_clean.csv')


    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.5, random_state=1)
    return X_train, X_test, y_train, y_test
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()

    print('Method  \tTrain\tTest\n'+'-'*30)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print('R2  linrg:\t%1.4f\t%1.4f'%(r2_score(y_train, linreg.predict(X_train)),r2_score(y_test, linreg.predict(X_test))))
    print('MSE linrg:\t%1.4f\t%1.4f\n'% (metrics.mean_squared_error(y_train, linreg.predict(X_train)), metrics.mean_squared_error(y_test, linreg.predict(X_test))))

    ridgereg = Ridge(alpha=0.001, normalize=True)
    ridgereg.fit(X_train, y_train)
    print('R2 ridge:\t%1.4f\t%1.4f'%(r2_score(y_train, ridgereg.predict(X_train)),r2_score(y_test, ridgereg.predict(X_test))))
    print('MSE linrg:\t%1.4f\t%1.4f\n' % (metrics.mean_squared_error(y_train, ridgereg.predict(X_train)),metrics.mean_squared_error(y_test, ridgereg.predict(X_test))))
    ridgeregcv = RidgeCV(alphas=10. ** numpy.arange(-4, 5), normalize=True, scoring='neg_mean_squared_error')

    ridgeregcv.fit(X_train, y_train)
    print('R2 rdgCV:\t%1.4f\t%1.4f' % (r2_score(y_train, ridgeregcv.predict(X_train)), r2_score(y_test, ridgeregcv.predict(X_test))))
    print('MSE linrg:\t%1.4f\t%1.4f\n' % (metrics.mean_squared_error(y_train, ridgeregcv.predict(X_train)),metrics.mean_squared_error(y_test, ridgeregcv.predict(X_test))))


    lassoreg = LassoCV(n_alphas=100, normalize=True, random_state=1,cv=3)
    lassoreg.fit(X_train, y_train)
    print('R2 lass:\t%1.4f\t%1.4f' % (r2_score(y_train, lassoreg.predict(X_train)), r2_score(y_test, lassoreg.predict(X_test))))
    print('MSE linrg:\t%1.4f\t%1.4f\n'% (metrics.mean_squared_error(y_train, lassoreg.predict(X_train)), metrics.mean_squared_error(y_test, lassoreg.predict(X_test))))