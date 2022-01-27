import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_feature_importance
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def ex_titanic():

    df,idx_target = pd.read_csv(folder_in + 'dataset_titanic.csv', delimiter='\t'),0
    df.drop(labels = ['alive'], axis = 1, inplace = True)
    FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
    print(FI.sort_values(by=FI.columns[0], ascending=False).to_string(index=False))
    for t in ['F_score','R2','C','XGB','SHAP','I']:
        P.plot_hor_bars(FI[t].to_numpy(), FI['features'].to_numpy(), legend=t,filename_out='FI_%s.png'%t)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():

    df,idx_target = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=','),2
    FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
    print(FI.sort_values(by=FI.columns[0], ascending=False).to_string(index=False))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_fuel():

    df,idx_target = pd.read_csv(folder_in + 'dataset_fuel.csv', delimiter='\t'),0
    df.drop(labels=['name'], axis=1, inplace=True)
    FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
    print(FI.sort_values(by=FI.columns[0], ascending=False).to_string(index=False))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_apnea():

    df,idx_target = pd.read_csv(folder_in + 'dataset_apnea.csv', delimiter='\t'),0
    FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
    print(FI.sort_values(by=FI.columns[0], ascending=False).to_string(index=False))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_flights():

    df,idx_target = pd.read_csv(folder_in+'dataset_kibana_flights.csv', sep=','),10
    FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
    print(FI.sort_values(by=FI.columns[0], ascending=False).to_string(index=False))
    for t in ['F_score','R2','C','XGB','SHAP','I']:
        P.plot_hor_bars(FI[t].to_numpy(), FI['features'].to_numpy(), legend=t,filename_out='FI_%s.png'%t)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_apnea()
    ex_titanic()
    #ex_flights()