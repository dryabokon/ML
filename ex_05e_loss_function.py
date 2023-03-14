import cv2
import pandas as pd
import numpy as numpy
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex1():
    x = log_loss(["A", "B", "C", "D"], [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    print(x)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex2():
    df, idx_target = pd.read_csv(folder_in + 'dataset_titanic.csv', sep='\t'), 0
    tools_DF.get_entropy(df, idx_target=0, idx_c1=1, idx_c2=2)
    return
# ----------------------------------------------------------------------------------------s------------------------------
def ex3():
    df, idx_target = pd.read_csv(folder_in + 'dataset_titanic.csv', sep='\t'), 0
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    columns = df.columns
    target = columns[idx_target]

    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)

    for i1 in range(len(idx) - 1):
        for i2 in range(i1 + 1, len(idx)):
            c1, c2 = columns[idx[i1]], columns[idx[i2]]
            I = tools_DF.get_Mutual_Information(df,idx_target,idx[i1],idx[i2])
            #I = mutual_info_classif(df[[c1, c2]], df[target]).sum()
            print(c1,c2,I)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex3()