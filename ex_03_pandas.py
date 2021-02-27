import cv2
import numpy
import pandas as pd
from sklearn.impute import MissingIndicator
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
# TODO Applying functions to the data
# TODO Merge
# TODO Grouping
# TODO Reshaping
# TODO Time series
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_01_create_data_series():
    s = pd.Series([1, 3, 5, numpy.nan, 6, 8])

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_01_create_data_frame_TS():
    rows, cols = 10, 3
    idx_dates = pd.date_range("20210101", periods=rows)
    columns = [chr(ord('A') + c) for c in range(cols)]
    A = (99 * numpy.random.random((rows, cols))).astype(int)
    df = pd.DataFrame(data=A, index=idx_dates, columns=columns)

    return df


# ----------------------------------------------------------------------------------------------------------------------
def ex_01_create_data_frame_meal():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 3, 1000),
         ('Lemon ', 9, 7000),
         ('Milk  ', 7, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    df = pd.DataFrame(data=A, index=None, columns=['Product', '#', 'Price'])
    df = df.astype({'#': 'int32','Price': 'int32'})


    return df


# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_quick(df):
    print('--------HEAD--------')
    print(df.head())
    print('\n--------TAIL--------')
    print(df.tail())

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_body(df):
    print(df)
    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_body_no_index(df):
    print(df.to_string(index=False))
    return


# ----------------------------------------------------------------------------------------------------------------------


def ex_02_inspect_columns(df):
    columns = df.columns.to_numpy()
    print(columns)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_index(df):
    idx = df.index.to_numpy() # str

    idx2 = (pd.to_datetime(idx).strftime('%y-%m-%d')).to_numpy()    # datetime
    for each in idx2:
        print(each)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_remove(df):
    df.drop(labels=['alive'], axis=1, inplace=True)
    return df
# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_columns(df):
    df_sliced1 = df[['Product', '#']]
    print(df_sliced1)
    print()

    df_sliced2 = df.loc[:,['Product', '#']]
    print(df_sliced2)
    print()

    df_sliced3 = df.iloc[:, [0,1]]
    print(df_sliced3)
    print()

    df_sliced4 = df.iloc[:, 0:2]
    print(df_sliced4)
    print()

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_rows(df):
    df_sliced1 = df[2:4]
    print(df_sliced1)
    print()

    df_sliced2 = df.loc[2:4]
    print(df_sliced2)
    print()

    df_sliced3 = df.iloc[2:4]
    print(df_sliced3)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_rows_v2(df):
    time_range = df.index.to_numpy()
    columns = df.columns.to_numpy()
    columns_filtered = columns[[0,1]]

    print('\n'+'-' * 32 + '\nslice over specific time')
    print(df.loc[time_range[:3], :])

    print('\n'+'-'*32 + '\nslice over selected columns')
    print(df.loc[:, columns_filtered])

    print('\n'+'-' * 32 + '\nslice over specific time and columns')
    print(df.loc[time_range[1:3], columns_filtered])

    print('\n'+'-' * 32 + '\nslice over specific time and columns')
    print(df.iloc[1:3, [0,1]])

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_08_order(df):
    print(df.sort_values(by='Product'))
    print('\n--------------------')
    print(df.sort_values(by='#', ascending=False))
    print('\n--------------------')
    print(df.sort_values(by='Price', ascending=False))

    return


# ----------------------------------------------------------------------------------------------------------------------

def ex_09_aggregates(df, idx_agg=0):

    col_label = df.columns.to_numpy()[idx_agg]
    df_agg = df.groupby(col_label).sum()
    print(df_agg)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_09_interpolate():
    S = [1, 3, 5, numpy.nan, 6, 8]
    S_avg = pd.Series(S).interpolate().values
    return


# ----------------------------------------------------------------------------------------------------------------------

def ex_10_IO_read():
    df = pd.read_csv('A.txt', sep='/t')
    A_numpy = df.values

    return
# ----------------------------------------------------------------------------------------------------------------------

def ex_10_IO_write(df):
    df.to_csv(folder_out + 'temp.csv', index=False, sep='\t')
    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_12_is_null(df):
    cv2.imwrite(folder_out + 'nans.png', 255 * (df.isnull()).to_numpy())

    return

# ----------------------------------------------------------------------------------------------------------------------
def ex_13_is_missing(df):
    indicator = MissingIndicator(missing_values=numpy.nan)
    res = indicator.fit_transform(df)
    cv2.imwrite(folder_out + 'missing.png', 255 * res)
    return

# ----------------------------------------------------------------------------------------------------------------------
def ex_14_hash(df):
    print(df[['sex']].head())
    print()
    sex = {'male': 0, 'female': 1}
    df['sex'] = df['sex'].map(sex)
    print(df[['sex']].head())
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_14_hash_categoricasl(df):
    df = df.dropna()
    print(df.head())
    print()
    df = tools_DF.hash_categoricals(df)
    print(df.head())
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_01_create_data_series()
    df_ts = ex_01_create_data_frame_TS()
    df_meal = ex_01_create_data_frame_meal()
    df_titanic = pd.read_csv(folder_in + 'dataset_titanic.csv', delimiter='\t')

    #ex_02_inspect_quick(df_meal)
    # ex_02_inspect_body(df_meal)
    # ex_02_inspect_body_no_index(df_meal)
    # ex_02_inspect_columns(df_meal)
    # ex_02_inspect_index(df_ts)


    #ex_07_slicing_columns(df_meal)
    #ex_07_slicing_rows(df_meal)
    #ex_07_slicing_rows_v2(df_ts)

    # ex_08_order(df_meal)
    # ex_09_aggregates(df_meal, 0)
    # ex_09_interpolate()
    # ex_10_IO_read()
    # ex_12_is_null(df_titanic)
    # ex_13_is_missing(df_titanic)
    # ex_14_hash(df_titanic)
    # ex_14_hash_categoricasl(df_titanic)







