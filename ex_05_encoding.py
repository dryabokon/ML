import cv2
import numpy as numpy
import pandas as pd
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_is_missing(df):
    A = (df.isnull()).to_numpy()
    B = MissingIndicator(missing_values=numpy.nan).fit_transform(df)
    cv2.imwrite(folder_out + 'nans_1.png', 255 * A)
    cv2.imwrite(folder_out + 'nans_2.png', 255 * B)
    print(df)
    print()
    print(A)
    print()
    print(B)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_replace(df):
    dct_replace = {numpy.NaN: 999.0}

    print(df)
    print()
    df.replace(dct_replace, inplace=True)
    print(df)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_imputer(df):
    imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')

    print(df)
    print()
    df = imp.fit_transform(df)
    print(df)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_imputer2(df):

    print(df)
    print()
    df.fillna(df.mean(), inplace=True)
    print(df)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_ordinal_01():
    df = pd.DataFrame(numpy.array([['M', 'O-', 'medium'],['M', 'O-', 'high'],['F', 'O+', 'high'],['F', 'AB', 'low'],['F', 'B+', numpy.nan]]))
    df.columns = ['sex', 'blood_type', 'edu_level']
    encoder = OrdinalEncoder()

    print(df)
    print()
    df.iloc[:,2] = encoder.fit_transform(df.iloc[:, 2].values.reshape((-1, 1)))
    print(df)

    # drawback missing value is encoded as a separate class
    # order of data is not respected

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_ordinal_02():
    df = pd.DataFrame(numpy.array([['M', 'O-', 'medium'], ['M', 'O-', 'high'], ['F', 'O+', 'high'], ['F', 'AB', 'low'], ['F', 'B+', numpy.nan]]))
    df.columns = ['sex', 'blood_type', 'edu_level']

    print(df)
    print()

    cat = pd.Categorical(df['edu_level'], categories=['missing', 'low', 'medium', 'high'], ordered=True)
    cat.fillna('missing')

    labels, unique = pd.factorize(cat, sort=True)
    df.edu_level = labels
    print(df)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_OneHot():
    df = pd.DataFrame(numpy.array([['M', 'O-', 'medium'], ['M', 'O-', 'high'], ['F', 'O+', 'high'], ['F', 'AB', 'low'], ['F', 'B+', numpy.nan]]))
    df.columns = ['sex', 'blood_type', 'edu_level']
    onehot = OneHotEncoder(dtype=numpy.int, sparse=True)

    df2 = pd.DataFrame(onehot.fit_transform(df[['sex', 'blood_type']]).toarray())
    df2.columns = numpy.unique(df['sex']).tolist() + numpy.unique(df['blood_type']).tolist()

    df2['edu_level'] = df.edu_level
    print(df2.to_string(index=False))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = pd.DataFrame(numpy.array([5, 7, 8, numpy.NaN, numpy.NaN, numpy.NaN, -5, 0, 25, 999, 1, -1, numpy.NaN, 0, numpy.NaN]).reshape((5, 3)))
    #ex_is_missing(df)
    #ex_replace(df)
    #ex_imputer2(df)
    # ex_encode_ordinal_01()
    # ex_encode_ordinal_02()
    ex_encode_OneHot()
