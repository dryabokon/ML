import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_feature_correlation(df):

    df = tools_DF.hash_categoricals(df)
    columns = df.columns.to_numpy()
    corrmat = abs(df.corr()).to_numpy()

    for i in range(corrmat.shape[0]):corrmat[i,i]=0

    ranks = []
    while len(ranks)<corrmat.shape[1]:
        idx = numpy.argmax(corrmat)
        r,c = numpy.unravel_index(idx,corrmat.shape)
        corrmat[r, c] = 0
        if r not in ranks:
            ranks.append(r)
        if c not in ranks:
            ranks.append(c)

    ranks = numpy.array(ranks)

    corrmat = abs(df[columns[ranks]].corr())

    for i in range(corrmat.shape[0]):
        corrmat.iloc[i,i]=numpy.nan


    plt.figure(figsize=(12, 8))
    sns.heatmap(corrmat, vmax=1, square=True, annot=True, fmt='.2f', cmap='GnBu', cbar_kws={"shrink": .5},robust=True)
    plt.savefig(folder_out+'corr.png')

    return
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


    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    #df = pd.read_csv(folder_in + 'dataset_tips.txt', delimiter='\t')
    #df = pd.read_csv(folder_in + 'dataset_titanic.txt', delimiter='\t')

    #ex_feature_correlation(df)
    ex_VIF(df)
    ex_VIF2(df)
