import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
# TODO Missing data
# TODO Applying functions to the data
# TODO Merge
# TODO Grouping
# TODO Reshaping
# TODO Time series
# TODO IO
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

def ex_02_inspect_header(df):
    print(df.columns.to_numpy())

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_02_inspect_index(df):
    idx = df.index.to_numpy()
    idx2 = (pd.to_datetime(idx).strftime('%y-%m-%d')).to_numpy()
    for each in idx2:
        print(each)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_columns(df):
    df_product = df[['Product', '#']]
    # print(df_product)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_rows(df):
    df_product_first_records = df[0:2]
    # print(df_product_first_records)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing_rows_v2(df):
    time_range = df.index.to_numpy()
    columns = df.columns.to_numpy()
    columns= columns[[0,1]]

    # slice over specific time
    print(df.loc[time_range[:3], :])

    # slice over selected columns
    print('\n--------------------')
    print(df.loc[:, columns])

    # slice over specific time and columns
    print('\n--------------------')
    print(df.loc[time_range[1:3], columns])

    print('\n--------------------')
    # slice over specific time and columns
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

def ex_09_aggregates(df, idx_agg):

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

def ex_10_IO():
    A = pd.read_csv('A.txt', sep='/t').values
    return


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_01_create_data_series()
    df_ts = ex_01_create_data_frame_TS()
    df_meal = ex_01_create_data_frame_meal()
    # ex_02_inspect_body(df)
    # ex_02_inspect_header(df)
    # ex_02_inspect_index(df)
    # ex_08_order(df)
    # ex_07_slicing_columns(df)
    # ex_07_slicing_rows(df)
    # ex_07_slicing_rows_v2(df_ts)
    ex_09_aggregates(df_meal, 0)

    #df.drop(labels = ['age', 'deck'], axis = 1, inplace = True)
    # hashing
    # sex = {'male': 0, 'female': 1}
    # data['sex'] = data['sex'].map(sex)
