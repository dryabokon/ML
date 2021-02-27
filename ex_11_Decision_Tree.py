from dtreeviz.trees import dtreeviz
import cv2
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import classifier_DTree
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_view_tree(df,idx_target):
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    X, Y = tools_DF.df_to_XY(df, idx_target, keep_categoirical=False)

    columns = df.columns.to_numpy()
    idx = numpy.delete(numpy.arange(0, len(columns)), idx_target)
    columns = columns[idx]

    C = classifier_DTree.classifier_DT(max_depth=3,folder_out=folder_out)
    C.learn(X, Y,columns,do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():
    df, idx_target = pd.read_csv(folder_in + 'dataset_titanic.csv', sep='\t'), 0
    df.drop(labels=['alive','deck'], axis=1, inplace=True)
    ex_view_tree(df,idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_income():
    df, idx_target = pd.read_csv(folder_in + 'dataset_income.csv', sep='\t'), 0
    #mean = df.iloc[:,idx_target].mean()
    th = df.iloc[:,idx_target].quantile(0.8)

    df.iloc[:,idx_target]=1*(df.iloc[:,idx_target]>th)
    ex_view_tree(df,idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_income()



