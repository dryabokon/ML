import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from sklearn import linear_model
import shap
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'dataset_titanic.csv', delimiter='\t'), 0
df.drop(labels=['alive'], axis=1, inplace=True)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df, idx_target = pd.read_csv(folder_in + 'dataset_apnea.csv', delimiter='\t'), 0

    #df = tools_DF.hash_categoricals(df)
    X,Y = tools_DF.df_to_XY(df,idx_target,numpy_style=False)
    # model = XGBRegressor().fit(X, Y)
    # explainer = shap.Explainer(model)

    model = linear_model.LogisticRegression().fit(X, Y)
    explainer = shap.LinearExplainer(model, X)

    shap_values = explainer(X)

    #shap.plots.waterfall(shap_values[0])
    shap.plots.force(shap_values[0])


    shap.plots.beeswarm(shap_values)