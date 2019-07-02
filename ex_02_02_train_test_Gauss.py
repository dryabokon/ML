import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
import classifier_Gauss
import classifier_Gauss2
import classifier_Gauss_indep
# ---------------------------------------------------------------------------------------------------------------------
def classify_data(filename_data_pos,filename_data_neg,folder_out):

    filename_scrs_pos = folder_out+'scores_pos.txt'
    filename_scrs_neg = folder_out+'scores_neg.txt'

    Classifier1 = classifier_Gauss.classifier_Gauss()
    Classifier2 = classifier_Gauss2.classifier_Gauss2()
    Classifier3 = classifier_Gauss_indep.classifier_Gauss_indep()


    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)
    ML = tools_ML.tools_ML(Classifier3)
    ML.E2E_features_2_classes(folder_out, filename_data_pos, filename_data_neg,filename_scrs_pos=filename_scrs_pos, filename_scrs_neg=filename_scrs_neg,fig=fig)

    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = 'data/ex_pos_neg_gauss/'
    folder_out = 'data/output/'
    classify_data(folder_in+'pos.txt', folder_in+'neg.txt',folder_out)
