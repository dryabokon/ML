import numpy as numpy
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_ordinal_01():
    df = pd.DataFrame(numpy.array([['M', 'O-', 'medium'],
                                   ['M', 'O-', 'high'],
                                   ['F', 'O+', 'high'],
                                   ['F', 'AB', 'low'],
                                   ['F', 'B+', numpy.nan]]))
    df.columns = ['sex', 'blood_type', 'edu_level']
    encoder = OrdinalEncoder()

    print(df)
    print()
    df.iloc[:,2] = encoder.fit_transform(df.iloc[:, 2].values.reshape((-1, 1)))
    print(df)

    # drawback: missing value is encoded as a separate class
    # drawback: order of data is not respected

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_ordinal_02():
    df = pd.DataFrame(numpy.array([['M', 'O-', 'medium'],
                                   ['M', 'O-', 'high'],
                                   ['F', 'O+', 'high'],
                                   ['F', 'AB', 'low'],
                                   ['F', 'B+', numpy.nan]]))
    df.columns = ['sex', 'blood_type', 'edu_level']

    print(df)
    print()

    cat = pd.Categorical(df['edu_level'],
                         categories=['missing', 'low', 'medium', 'high'], ordered=True)
    cat.fillna('missing')

    labels, unique = pd.factorize(cat, sort=True)
    df.edu_level = labels
    print(df)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_encode_OneHot():
    df = pd.DataFrame(numpy.array([['M', 'O-'],
                                   ['M', 'O-'],
                                   ['F', 'O+'],
                                   ['F', 'AB'],
                                   ['F', 'B+']]))
    df.columns = ['sex', 'blood_type']
    onehot = OneHotEncoder(dtype=numpy.int, sparse=True)

    df2 = pd.DataFrame(onehot.fit_transform(df[['sex', 'blood_type']]).toarray())
    df2.columns = numpy.unique(df['sex']).tolist() + numpy.unique(df['blood_type']).tolist()


    print(df.to_string(index=False))
    print()
    print(df2.to_string(index=False))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_encode_ordinal_01()
    #ex_encode_ordinal_02()
    ex_encode_OneHot()
