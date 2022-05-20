import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
import tools_feature_importance
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_TS/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def plot_all_in_one(df0, idx_target):

    df0 = tools_DF.hash_categoricals(df0)
    #df0 = tools_DF.scale(df0)

    target = df0.columns[idx_target]
    features = df0.columns.to_numpy()[numpy.delete(numpy.arange(0, df0.shape[1]), idx_target)]
    FI = tools_feature_importance.feature_imporance_F_score(df0, idx_target)

    best_idx = numpy.argsort(-FI)
    best_features = features[best_idx][:4]
    df = df0[[target]+best_features.tolist()]

    df = tools_DF.hash_categoricals(df)
    P.TS_seaborn(df,numpy.arange(1,df.shape[1]).tolist(),None,filename_out='all_best_features.png')
    P.TS_seaborn(df,0,None,filename_out='target.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = tools_DF.hash_categoricals(df)

    #P.plot_TS_separatly(df, idx_target)
    plot_all_in_one(df, idx_target)
    #P.plot_target_feature(df, idx_target)
