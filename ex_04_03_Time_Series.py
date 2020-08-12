#http://pandas.pydata.org/pandas-docs/version/0.15/10min.html
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import math
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# ----------------------------------------------------------------------------------------------------------------------
import generator_TS
# ----------------------------------------------------------------------------------------------------------------------
import TS_Naive
import TS_LinearRegression
import TS_AutoRegression
import TS_Holt
import TS_VAR
#import TS_LSTM

# ---------------------------------------------------------------------------------------------------------------------
import tools_TS
import tools_IO
import tools_plot
# ---------------------------------------------------------------------------------------------------------------------
def generate_data():
    G = generator_TS.generator_TS()
    G.generate_linear('data/ex50/YX_linear.txt')
    G.generate_sine('data/ex50/YX_sine.txt')
    return
# ---------------------------------------------------------------------------------------------------------------------
def predict_series(filename_series, folder_out, target_column,delim,fit_only):

    C1 = TS_Naive.TS_Naive()
    C2 = TS_LinearRegression.TS_LinearRegression()
    C3 = TS_AutoRegression.TS_AutoRegression()
    C4 = TS_Holt.TS_Holt()
    C5 = TS_VAR.TS_VAR()
    #C6 = TS_LSTM.TS_LSTM()

    TS = tools_TS.tools_TS(C4,delim=delim)

    if fit_only==True:
        TS.E2E_fit(filename_series, folder_out, target_column=target_column, verbose=True)
    else:
        TS.E2E_train_test(filename_series, folder_out, target_column=target_column,verbose=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_predictors(filename_series, folder_out,target_column,delim, max_len,fit_only):

    C1 = TS_Naive.TS_Naive()
    C2 = TS_LinearRegression.TS_LinearRegression()
    C3 = TS_AutoRegression.TS_AutoRegression()
    C4 = TS_Holt.TS_Holt()
    C5 = TS_VAR.TS_VAR()

    Classifiers = [C1,C2,C3,C4,C5]
    list_filenames,list_train_filenames,list_test_filenames = [],[],[]

    suffix =''
    if target_column >= 0:suffix = ('%03d_' % target_column)
    print(filename_series.split('/')[-1])

    for i in range(0,len(Classifiers)):
        TS = tools_TS.tools_TS(Classifiers[i],delim,max_len)
        if fit_only == True:
            TS.E2E_fit(filename_series, folder_out, target_column=target_column, verbose=False)
            filename_fact = folder_out + 'fact' + suffix + '.txt'
            filename_pred = folder_out + 'fit_' + suffix + Classifiers[i].name + '.txt'
            list_filenames.append(filename_pred)
            smape = TS.calc_SMAPE_for_files(filename_fact, filename_pred)
            print('SMAPE Fit = %1.2f - %s' % (smape,Classifiers[i].name))
        else:
            TS.E2E_train_test(filename_series, folder_out, target_column=target_column, verbose=False)
            filename_train_fact = folder_out + 'train_fact' + suffix + '.txt'
            filename_test_fact  = folder_out + 'test_fact'  + suffix + '.txt'
            filename_pred_train = folder_out + 'train_pred_' + suffix + Classifiers[i].name + '.txt'
            filename_pred_test  = folder_out + 'test_pred_'  + suffix + Classifiers[i].name + '.txt'
            list_train_filenames.append(filename_pred_train)
            list_test_filenames.append (filename_pred_test)
            smape_train = TS.calc_SMAPE_for_files(filename_train_fact, filename_pred_train)
            smape_test  = TS.calc_SMAPE_for_files(filename_test_fact , filename_pred_test)
            print('SMAPE Train Test %01.02f\t%1.2f - %s' % (smape_train,smape_test,Classifiers[i].name))

    return
# ----------------------------------------------------------------------------------------------------------------------
filename_in = 'electricity_hourly_small.txt'
delim = ','
# ----------------------------------------------------------------------------------------------------------------------
#filename = 'YX_sine.txt'
#delim = '\t'
# ----------------------------------------------------------------------------------------------------------------------
#filename = 'Monthly-test.txt'
#delim = '\t'
# ----------------------------------------------------------------------------------------------------------------------
def encoder_labelencoder():
    X = numpy.array([['Male ', 4], ['Fmale', 1], ['Fmale', 2]])

    encoder = LabelEncoder()
    encoder.fit(X[:,0])
    XX = numpy.array(X)
    XX[:,0]=encoder.fit_transform(X[:,0])

    for x,xx in zip(X,XX):print(x,'=',xx)

    return
# ----------------------------------------------------------------------------------------------------------------------
def encoder_OneHotEncoder():
    X = [['Male ', 4], ['Fmale', 1], ['Fmale', 2]]

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    XX = numpy.array(enc.transform(X).toarray()).astype(int)


    for x,xx in zip(X,XX):print(x,'=',xx)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = 'data/ex_TS/'
    folder_out = 'data/output/'

    tools_IO.remove_files(folder_out, create=True)
    predict_series(folder_in + filename_in, folder_out,target_column=1,delim=delim,fit_only=True)
    #benchmark_predictors(folder_in + filename, folder_out,target_column=1,delim=delim,max_len=2000,fit_only=False)

