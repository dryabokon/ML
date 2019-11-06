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
import ex_07_04_audio_features
# ---------------------------------------------------------------------------------------------------------------------
def E2E_features_multi_classes(folder_in,folder_out,has_header,has_labels_first_col):

    Classifier = classifier_RF.classifier_RF()

    ML = tools_ML.tools_ML(Classifier)
    ML.E2E_features(folder_in, folder_out,has_header=has_header,has_labels_first_col=has_labels_first_col)

    return
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = False, True
#folder_in = 'data/ex_pos_neg_linear/'
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = True, True
#folder_in = 'data/ex_pos_neg_apnea/'
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = False, True
#folder_in = 'data/ex_features_LPR/CNN_App_Keras/'
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = True, True
#folder_in = 'data/ex_features_digits_mnist/CNN_AlexNet_TF/'
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = False, True
folder_in = 'data/ex_features_sounds/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in  = './data/output_sounds/'
    folder_out = './data/output/'
    ex_07_04_audio_features.extract_features_from_sound_folder(folder_in,folder_out)
    E2E_features_multi_classes(folder_out,folder_out,has_header,has_labels_first_col)


