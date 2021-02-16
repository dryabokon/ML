import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df, idx_target = pd.read_csv(folder_in + 'dataset_survey.txt', delimiter='\t'), 0
    X,Y = tools_DF.df_to_XY(df,idx_target,keep_categoirical=False,drop_na=True)
    columns = df.columns.to_numpy()[numpy.delete(numpy.arange(0, df.shape[1]), idx_target)]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    categorical_columns = ['RACE', 'OCCUPATION', 'SECTOR','MARR', 'UNION', 'SEX', 'SOUTH']
    numerical_columns = ['EDUCATION', 'EXPERIENCE', 'AGE']
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_columns),remainder='passthrough')
    model = make_pipeline(preprocessor,TransformedTargetRegressor(regressor=Ridge(alpha=1e-10),func=numpy.log,inverse_func=numpy.exp))
    model.fit(X_train, Y_train)

    # ridgereg = linear_model.Ridge(alpha=0.001, normalize=True)
    # ridgereg.fit(X_train, Y_train)
    # values = ridgereg.coef_ * X_train.std(axis=0)
    # #values = X_train.std(axis=0)
    #
    # idx = numpy.argsort(-values)
    #
    # for feature_name, value in zip(columns[idx], values[idx]):
    #     print('%1.2f\t%s' % (value, feature_name))


