import numpy
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_ML_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
ML = tools_ML_v2.ML(None,folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_VIF(df):

    df = tools_DF.hash_categoricals(df)
    df = df.dropna()
    columns = df.columns.to_numpy()
    VIFs = numpy.array([variance_inflation_factor(df.values, i) for i in range(df.shape[1])])
    idx = numpy.argsort(VIFs)

    for i in idx:
        print('%1.2f\t%s'%(VIFs[i],columns[i]))

    return

# ----------------------------------------------------------------------------------------------------------------------
def ex_VIF2(df):

    df = tools_DF.hash_categoricals(df)
    df = df.dropna()
    columns = df.columns
    VIFs = []

    for i in range(0, columns.shape[0]):
        y = df[columns[i]]
        x = df[columns.drop([columns[i]])]
        r2 = OLS(y, x).fit().rsquared
        vif = round(1 / (1 - r2), 2)
        VIFs.append(vif)

    idx = numpy.argsort(VIFs)
    for i in idx:
        print('%1.2f\t%s' % (VIFs[i], columns[i]))

    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    #df = pd.read_csv(folder_in + 'dataset_tips.txt', delimiter='\t')
    df = pd.read_csv(folder_in + 'dataset_titanic.csv', delimiter='\t')

    ML.feature_correlation(df)

