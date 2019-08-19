#http://pandas.pydata.org/pandas-docs/version/0.15/10min.html
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import math
import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import generator_TS
# ----------------------------------------------------------------------------------------------------------------------
import TS_Naive
import TS_LinearRegression
import TS_AutoRegression
import TS_Holt
import TS_VAR
import TS_LSTM
import TS_ARIMA
# ---------------------------------------------------------------------------------------------------------------------
import tools_TS
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
def generate_data():
    G = generator_TS.generator_TS()
    G.generate_linear('data/ex50/YX_linear.txt')
    G.generate_sine('data/ex50/YX_sine.txt')
    return
# ---------------------------------------------------------------------------------------------------------------------
def predict_series(filename_series, folder_out, target_column, fit_only):


    C1 = TS_Naive.TS_Naive()
    C2 = TS_LinearRegression.TS_LinearRegression()
    C3 = TS_AutoRegression.TS_AutoRegression()
    C4 = TS_Holt.TS_Holt()
    C5 = TS_VAR.TS_VAR()
    C6 = TS_LSTM.TS_LSTM()

    TS = tools_TS.tools_TS(C1)

    if fit_only==True:
        TS.E2E_fit(filename_series, folder_out, target_column=target_column, verbose=True)
    else:
        TS.E2E_train_test(filename_series, folder_out, target_column=target_column,verbose=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_predictors(filename_series, folder_out,target_column, fit_only):

    tools_IO.remove_files(folder_out, create=True)

    C1 = TS_Naive.TS_Naive()
    C2 = TS_LinearRegression.TS_LinearRegression()
    C3 = TS_AutoRegression.TS_AutoRegression()
    C4 = TS_Holt.TS_Holt()
    C5 = TS_VAR.TS_VAR()

    Classifiers = [C1, C2, C3, C4, C5]
    list_filenames,list_train_filenames,list_test_filenames = [],[],[]
    filename_fact,filename_train_fact,filename_test_fact = 0,0,0

    suffix =''
    if target_column >= 0:suffix = ('%03d_' % target_column)
    print(filename_series.split('/')[-1])

    for i in range(0,len(Classifiers)):
        TS = tools_TS.tools_TS(Classifiers[i])
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

    if target_column>=0:
        if fit_only == True:
            tools_IO.plot_multiple_series(filename_fact, list_filenames, target_column,caption='Fit '+ filename_series.split('/')[-1])
        else:
            tools_IO.plot_multiple_series(filename_train_fact, list_train_filenames, target_column,caption='Train '+ filename_series.split('/')[-1])
            tools_IO.plot_multiple_series(filename_test_fact,  list_test_filenames,  target_column,caption='Test ' + filename_series.split('/')[-1])

    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = 'data/ex_TS/'
    folder_out = 'data/output/'

    filename = 'Monthly-test.txt'

    predict_series(folder_in + filename, folder_out,target_column=-1,fit_only=False)
    #benchmark_predictors(folder_in + filename, folder_out,target_column=-1,fit_only=True)


