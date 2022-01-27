import re
import datetime
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
def ex_01_create_data_frame_v2():
    df = pd.DataFrame({
        'a': [1, 1, 1],
        'b': [2, 2, 2],
        'c': [0, 1, 2],
    })

    return df

# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_quick(df):
    print('--------HEAD--------')
    print (df.to_string(index=False))
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
def ex_03_remove(df):
    df.drop(labels=['alive'], axis=1, inplace=True)
    return df
# ----------------------------------------------------------------------------------------------------------------------
def ex_04_append(df):
    idx_target = 0
    c_new = 5

    M = numpy.full((c_new,df.shape[1]),numpy.nan)
    M[:,idx_target] = numpy.linspace(0,c_new-1,c_new)
    df2 = pd.DataFrame(M,columns=df.columns)
    print(df)
    print('-'*10)
    df = df.append(df2,ignore_index=True)

    print(df)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_04_display_precision():
    A = numpy.array([[1.00002]])
    df = pd.DataFrame(A)
    print(df)
    print()

    pd.set_option("display.precision", 2)
    print(df)


    return
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
def ex_concat():
    delta = datetime.timedelta(minutes=0, seconds=1)
    start = datetime.datetime.strptime('2020-11-26 12:10:35', "%Y-%m-%d %H:%M:%S")
    idx1 = [start + delta * t for t in [0, 1, 2, 3]]
    idx4 = [start + delta * t for t in [2, 3, 6, 7]]

    df1 = pd.DataFrame(
    {
    "A": ["A0", "A1", "A2", "A3"],
    "B": ["B0", "B1", "B2", "B3"],
    "C": ["C0", "C1", "C2", "C3"],
    "D": ["D0", "D1", "D2", "D3"],
    },
    index = idx1)

    df4 = pd.DataFrame(
        {
        "B": ["B2", "B3", "B6", "B7"],
        "D": ["D2", "D3", "D6", "D7"],
        "F": ["F2", "F3", "F6", "F7"],
        },index = idx4)

    result = pd.concat([df1, df4], axis=1)

    print(result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_align():

    df1 = pd.DataFrame(
    {
    "A": ["A0", "A1", "A2"],
    "B": ["B0", "B1", "B2"],
    },
    index = [0,1,2])

    df2 = pd.DataFrame(
        {
        "B": ["B1", "B2", "B3"],
        "C": ["C1", "C2", "C3"],
        },index = [1,2,3])


    print(df1)
    print(df2)
    print()

    print('outer')
    a1, a2 = df1.align(df2, join='outer', axis=0)
    print(a1)
    print(a2)
    print()

    print('inner')
    a1, a2 = df1.align(df2, join='inner', axis=0)
    print(a1)
    print(a2)
    print()

    print('left')
    a1, a2 = df1.align(df2, join='left', axis=0)
    print(a1)
    print(a2)
    print()

    print('right')
    a1, a2 = df1.align(df2, join='right', axis=0)
    print(a1)
    print(a2)
    print()

    # xx = pd.concat([df1, df2], axis=1).iloc[:,df1.shape[1]:]
    # print(xx)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_merge():
    df_left = pd.DataFrame(
        {
            "key1": ["K0", "K0", "K1", "K2"],
            "key2": ["K0", "K1", "K0", "K1"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    df_right = pd.DataFrame(
        {
            "key1": ["K0", "K1", "K1", "K2"],
            "key2": ["K0", "K0", "K0", "K0"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    print(df_left)
    print()
    print(df_right)
    print('\n'+''.join(['-']*48))

    # left  - LEFT OUTER JOIN  - Use keys from left frame only
    # right - RIGHT OUTER JOIN - Use keys from right frame only
    # outer - FULL OUTER JOIN  - Use union of keys from both frames
    # inner - INNER JOIN Use   - intersection of keys from both frames

    how = 'right'
    key = 'key1'
    df_result = pd.merge(df_left, df_right, how=how, on=[key])
    print('{how} on {key}'.format(how=how,key=key))
    print(df_result)
    print('\n' + ''.join(['-'] * 48))

    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # ex_01_create_data_series()
    # df_ts = ex_01_create_data_frame_TS()
    # df_meal = ex_01_create_data_frame_meal()
    # df_titanic = pd.read_csv(folder_in + 'dataset_titanic.csv', delimiter='\t')

    # ex_02_inspect_quick(df_meal)
    # ex_02_inspect_body(df_meal)
    # ex_02_inspect_body_no_index(df_meal)
    # ex_02_inspect_columns(df_meal)
    # ex_02_inspect_index(df_ts)

    # ex_04_append(df_ts)
    # ex_04_display_precision()

    # ex_07_slicing_columns(df_meal)
    # ex_07_slicing_rows(df_meal)
    # ex_07_slicing_rows_v2(df_ts)

    # ex_08_order(df_meal)
    # ex_09_aggregates(df_meal, 0)
    # ex_09_interpolate()
    # ex_10_IO_read()
    # ex_12_is_null(df_titanic)
    # ex_13_is_missing(df_titanic)
    # ex_14_hash(df_titanic)
    # ex_14_hash_categoricasl(df_titanic)

    # ex_concat()
    #ex_align()
    ex_merge()

