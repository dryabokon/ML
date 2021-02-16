# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import classifier_KNN
#import classifier_SVM
import classifier_LM
import classifier_DTree
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
import tools_ML_enhanced
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_moon():
    C = classifier_KNN.classifier_KNN()
    df,target = pd.read_csv(folder_in+'dataset_moons.txt', sep='\t'),0
    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=target,idx_columns=[1,2])
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_crime():
    C = classifier_LM.classifier_LM()
    df,target = pd.read_csv(folder_in+'dataset_crime_clean.csv', sep=','),-1
    ML = tools_ML_v2.tools_ML_enhanced(C)
    ML.E2E_train_test_df(df,idx_target=target,idx_columns=[1,2])
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_crime()

