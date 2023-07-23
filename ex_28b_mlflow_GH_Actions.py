import os
import sys

import numpy.random
import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
sys.path.insert(1, './tools/')
# ---------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
import tools_DF
from classifier import classifier_LM
import tools_plot_v2
import tools_MLflower
import tools_IO
import tools_time_profiler
# ---------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ---------------------------------------------------------------------------------------------------------------------
# host = '34.122.102.32'
# port = '5000'
# remote_username = 'rsa-key-20200330'
# ppk_key_path = 'C:/Users/Anna/.ssh/ssh_rsa_dima_OpenSSH'
# F = tools_MLflower.MLFlower(host,port,remote_storage_folder='~/sources/ex_mlflow',remote_username=remote_username, ppk_key_path=ppk_key_path)
# ---------------------------------------------------------------------------------------------------------------------
host = os.environ.get("SECRET_HOST")
port = os.environ.get("SECRET_PORT")
remote_username = os.environ.get("SECRET_USERNAME")
ppk_value = os.environ.get("SECRET_PPK_VALUE")
ppk_key_path = './private_key'
with open(ppk_key_path, mode='w', encoding='utf-8') as f:
    f.write(ppk_value)
os.system('chmod 400 %s'%ppk_key_path)
F = tools_MLflower.MLFlower(host,port,remote_storage_folder='~/sources/ex_mlflow',remote_username=remote_username, ppk_key_path=ppk_key_path)
# ---------------------------------------------------------------------------------------------------------------------
C = classifier_LM.classifier_LM()
P = tools_plot_v2.Plotter(folder_out)
ML = tools_ML_v2.ML(C, folder_out=folder_out)

T = tools_time_profiler.Time_Profiler(verbose=False)
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv(folder_in + 'dataset_titanic.csv', sep='\t')
    df.drop(columns=['alive', 'deck'], inplace=True)
    idx_target = df.columns.get_loc('survived')

    P.set_color(0, P.color_blue)
    P.set_color(1, P.color_amber)
    df = tools_DF.hash_categoricals(df)
    #df = df.iloc[numpy.random.choice(df.shape[0],int(0.75*df.shape[0]),replace=False)]
    T.tic('E2E_train_test_df')
    df_metrics = ML.E2E_train_test_df(df, idx_target=idx_target, do_charts=True, do_pca=True)
    duration = T.print_duration('E2E_train_test_df')
    print(tools_DF.prettify(df_metrics, showindex=False))

    metrics = {'F1 train': df_metrics.iloc[-1, 1], 'F1 test': df_metrics.iloc[-1, 2],'train time': int(duration.split(':')[0]) * 60 + int(duration.split(':')[1])}
    params = {}

    artifacts = [folder_out + f for f in tools_IO.get_filenames(folder_out,'*.png')]
    F.save_experiment(experiment_name='CI: integration tests',params=params,metrics=metrics,artifacts=artifacts)