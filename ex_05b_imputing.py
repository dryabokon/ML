import cv2
import numpy as numpy
import pandas as pd
from sklearn.impute import SimpleImputer
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter()
# ----------------------------------------------------------------------------------------------------------------------
def ex_is_missing(df):
    A = (df.isnull()).to_numpy()

    cv2.imwrite(folder_out + 'nans_1.png', 255 * A)
    print(df)
    print()
    print(A)


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
def ex_fillna(df):

    print(df)
    print()
    df.fillna(df.mean(), inplace=True)
    print(df)
    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_imputer(df,idx_column,do_verbose=False,do_debug=False,):


    df_res = df.iloc[:,[idx_column]].copy()

    strategies = ['mean', 'median', 'most_frequent', 'interpolate']
    for strategy in strategies:
        if strategy == 'interpolate':
            B = df.iloc[:,[idx_column]].interpolate()
        else:
            imp = SimpleImputer(missing_values=numpy.nan,strategy=strategy)
            B = pd.DataFrame(data=imp.fit_transform(df.iloc[:,[idx_column]]),
                             index=df.index,columns=[df.columns[idx_column]])

        df_res = pd.concat([df_res, B], axis=1)

    df_res.columns = ['original'] + strategies
    if do_verbose:
        print(df_res)
        print()

    if do_debug:
        idx = numpy.arange(0, len(strategies) + 1)
        P.TSs_seaborn(df_res, idxs_target=idx, idx_feature=None,filename_out='imputer_ALL.png')

        for i in idx[1:]:
            P.TSs_seaborn(df_res, idxs_target=[i, 0], idx_feature=None,filename_out='imputer_%s.png' % strategies[i-1])


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pd.set_option("display.precision", 2)

    df_electro = pd.read_csv(folder_in + 'dataset_electricity.csv', delimiter=',')[:100]
    idx = numpy.random.choice(df_electro.shape[0], int(df_electro.shape[0] * 0.6))

    df_electro.iloc[idx, 1] = numpy.nan

    df_dummy = pd.DataFrame(numpy.array([[5,9,-5,999,3],
                                         [7,numpy.NaN,0,1,0],
                                         [9,numpy.NaN,25,-1,numpy.NaN]]).T)

    #ex_is_missing(df_dummy)
    #ex_replace(df_dummy)
    #ex_fillna(df)

    #ex_imputer(df_dummy,idx_column=1,do_verbose=True,do_debug=False)
    ex_imputer(df_electro,idx_column=1,do_verbose=False,do_debug=True)


