import matplotlib.pyplot as plt
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
import classifier_LM
import classifier_SVM
# ---------------------------------------------------------------------------------------------------------------------
def classify_data(filename_data_pos,filename_data_neg,folder_out):

    filename_scrs_pos = folder_out+'scores_pos.txt'
    filename_scrs_neg = folder_out+'scores_neg.txt'

    Classifier1 = classifier_LM.classifier_LM()
    Classifier2 = classifier_SVM.classifier_SVM(kernel='linear')

    ML = tools_ML.tools_ML(Classifier1)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)
    ML.E2E_features_2_classes(folder_out, filename_data_pos, filename_data_neg,filename_scrs_pos=filename_scrs_pos, filename_scrs_neg=filename_scrs_neg,fig=fig)
    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = 'data/ex06/'
    folder_out = 'data/output/'
    classify_data(folder_in+'pos.txt', folder_in+'neg.txt',folder_out)
