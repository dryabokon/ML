import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import classifier_Bayes
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_Gauss
import classifier_LM
import classifier_KNN
# ---------------------------------------------------------------------------------------------------------------------
import generator_Bay
import generator_Gauss
# ---------------------------------------------------------------------------------------------------------------------
from ex_02_01_train_test_2_classes import generate_data_syntetic
# ---------------------------------------------------------------------------------------------------------------------
def E2E_features_2_classes_multi_dim(filename_data_pos,filename_data_neg,folder_out,has_header,has_labels_first_col):

    filename_scrs_pos = folder_out+'scores_pos.txt'
    filename_scrs_neg = folder_out+'scores_neg.txt'

    #Classifier = classifier_Bayes.classifier_Bayes()
    #Classifier = classifier_KNN.classifier_KNN()
    #Classifier = classifier_Ada.classifier_Ada()
    #Classifier = classifier_SVM.classifier_SVM()
    Classifier = classifier_RF.classifier_RF()
    #Classifier = classifier_LM.classifier_LM()
    #Classifier = classifier_Gauss.classifier_Gauss()

    ML = tools_ML.tools_ML(Classifier)
    ML.E2E_features_2_classes_multi_dim(folder_out, filename_data_pos, filename_data_neg,has_header,has_labels_first_col)

    return
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = False, True
folder_in = 'data/ex_pos_neg_linear/'
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = True, True
#folder_in = 'data/ex_pos_neg_apnea/'
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = False, True
#folder_in = 'data/ex_features_LPR/CNN_App_Keras/'
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = True, True
folder_in = 'data/ex_pos_neg_football/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    folder_out = 'data/output/'
    E2E_features_2_classes_multi_dim(folder_in+'pos.txt', folder_in+'neg.txt',folder_out,has_header,has_labels_first_col)


